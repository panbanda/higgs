// SPDX-License-Identifier: Apache-2.0
//! Disk-backed wrapper around the in-memory paged prefix cache.

use std::path::PathBuf;
use std::time::Duration;

use half::f16;
use higgs_models::AnyCache;
use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache, slice_axis2};
#[cfg(test)]
use higgs_models::dflash::DFlashSnapshot;
use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::cache::disk_storage::{
    DiskCacheBlock, DiskCacheEntryMetadata, DiskCacheError, DiskCacheFileHeader, DiskCacheLayer,
    DiskCacheSnapshot, DiskStorage,
};
use crate::cache::paired::{PairedCacheError, RadixPairCheckpoint};
use crate::paged_prefix_cache::{
    PagedPairedLookupPlan, PagedPrefixCache, PagedPrefixMatch, PairedPrefixCacheStats,
    PairedPrepareTicket, PairedTouchToken, PreparedPairedPrefix,
};

pub const DEFAULT_MIN_TOKENS_TO_PERSIST: usize = 512;
pub const DEFAULT_MAX_DISK_BLOCKS: usize = 4096;

#[derive(Debug, Clone)]
pub struct DiskPrefixCacheConfig {
    pub disk_path: PathBuf,
    pub max_disk_blocks: usize,
    pub min_tokens_to_persist: usize,
}

/// Prefix cache that mirrors durable dense KV snapshots to disk.
pub struct DiskPrefixCache {
    memory: PagedPrefixCache,
    storage: Option<DiskStorage>,
    block_size: usize,
    min_tokens_to_persist: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DiskPrefixCandidate {
    session_id: u64,
    prefix_len: usize,
}

impl DiskPrefixCandidate {
    pub const fn prefix_len(&self) -> usize {
        self.prefix_len
    }
}

impl DiskPrefixCache {
    pub fn memory_only(max_entries: usize, block_size: usize) -> Self {
        Self {
            memory: PagedPrefixCache::new(max_entries, block_size),
            storage: None,
            block_size,
            min_tokens_to_persist: usize::MAX,
        }
    }

    pub fn new(
        max_entries: usize,
        block_size: usize,
        config: DiskPrefixCacheConfig,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DiskCacheError> {
        let DiskPrefixCacheConfig {
            disk_path,
            max_disk_blocks,
            min_tokens_to_persist,
        } = config;
        let storage = DiskStorage::open(
            &disk_path,
            block_size,
            max_disk_blocks,
            num_kv_heads,
            head_dim,
        )?;
        Ok(Self {
            memory: PagedPrefixCache::new(max_entries, block_size),
            storage: Some(storage),
            block_size,
            min_tokens_to_persist,
        })
    }

    /// Find the longest matching prefix. The cheap memory lookup runs first; a
    /// disk snapshot is only materialized when its indexed metadata says it can
    /// beat that memory hit.
    pub fn find_longest_prefix(
        &mut self,
        tokens: &[u32],
        checkpoint_id: Option<&str>,
    ) -> Option<PagedPrefixMatch> {
        let memory_match = self.memory.find_longest_prefix(tokens);
        let min_disk_prefix_len = memory_match
            .as_ref()
            .map_or(0, |matched| matched.prefix_len.saturating_add(1));
        let disk_match = self
            .find_disk_prefix_candidate(tokens, checkpoint_id, min_disk_prefix_len)
            .and_then(|candidate| self.load_disk_prefix_candidate(tokens, candidate));
        match (disk_match, memory_match) {
            (Some(disk), Some(memory)) if memory.prefix_len > disk.prefix_len => Some(memory),
            (Some(disk), _) => Some(disk),
            (None, memory) => memory,
        }
    }

    pub fn find_memory_prefix(&mut self, tokens: &[u32]) -> Option<PagedPrefixMatch> {
        self.memory.find_longest_prefix(tokens)
    }

    /// Find paired target/dFlash state in this process only.
    ///
    /// dFlash snapshots are deliberately never persisted, so this method must
    /// not fall back to the target-only disk index.
    pub(crate) fn plan_memory_paired_prefix(
        &mut self,
        tokens: &[u32],
    ) -> Result<Option<PagedPairedLookupPlan>, PairedCacheError> {
        self.memory.plan_longest_paired_prefix(tokens)
    }

    /// Configure whole-endpoint idle expiry for memory-only target+dSpark
    /// pairs. Target-only memory and disk snapshots retain their existing
    /// policies.
    pub(crate) const fn set_paired_idle_ttl(&mut self, ttl: Option<Duration>) {
        self.memory.set_paired_idle_ttl(ttl);
    }

    /// Remove expired memory-only target+dSpark endpoints atomically.
    pub(crate) fn evict_idle_paired(&mut self) -> usize {
        self.memory.evict_idle_paired()
    }

    #[must_use]
    pub(crate) const fn paired_prepare_ticket(&self) -> PairedPrepareTicket {
        self.memory.paired_prepare_ticket()
    }

    pub(crate) fn touch_memory_paired(&mut self, token: PairedTouchToken) -> bool {
        self.memory.touch_paired(token)
    }

    pub fn find_disk_prefix_candidate(
        &self,
        tokens: &[u32],
        checkpoint_id: Option<&str>,
        min_prefix_len: usize,
    ) -> Option<DiskPrefixCandidate> {
        let storage = self.storage.as_ref()?;

        if let Some(checkpoint) = checkpoint_id {
            let checkpoint_session_id = hash_checkpoint_id(checkpoint);
            if let Some(metadata) = storage.snapshot_metadata(checkpoint_session_id) {
                if snapshot_metadata_matches(tokens, metadata, min_prefix_len) {
                    return Some(DiskPrefixCandidate {
                        session_id: checkpoint_session_id,
                        prefix_len: metadata.token_count,
                    });
                }
            }
        }

        for prefix in block_prefix_hashes(tokens, self.block_size).iter().rev() {
            if prefix.len < min_prefix_len {
                break;
            }
            let Some(metadata) = storage.snapshot_metadata(prefix.session_id) else {
                continue;
            };
            if metadata.token_count == prefix.len && metadata.token_hash == prefix.token_hash {
                return Some(DiskPrefixCandidate {
                    session_id: prefix.session_id,
                    prefix_len: prefix.len,
                });
            }
        }
        None
    }

    pub fn load_disk_prefix_candidate(
        &mut self,
        tokens: &[u32],
        candidate: DiskPrefixCandidate,
    ) -> Option<PagedPrefixMatch> {
        let matched = self.load_snapshot_match(tokens, candidate.session_id)?;
        if matched.prefix_len == candidate.prefix_len {
            Some(matched)
        } else {
            tracing::debug!(
                expected_prefix_len = candidate.prefix_len,
                found_prefix_len = matched.prefix_len,
                "Skipping disk prefix cache snapshot whose indexed length changed"
            );
            None
        }
    }

    /// Store in memory and, when large enough, append a dense f16 snapshot to
    /// disk. Unsupported cache shapes remain memory-only.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: &AnyCache, checkpoint_id: Option<&str>) {
        self.memory.store(prefix_tokens, cache);
        self.persist_target(prefix_tokens, cache, checkpoint_id);
    }

    /// Prepare paired state without borrowing the disk/prefix cache.
    ///
    /// First-release dSpark state is memory-only. Ordinary target-only stores
    /// retain their existing disk behavior through [`Self::store`].
    pub(crate) fn prepare_memory_paired_prefix(
        ticket: PairedPrepareTicket,
        checkpoint: RadixPairCheckpoint<'_>,
    ) -> Result<PreparedPairedPrefix, PairedCacheError> {
        PagedPrefixCache::prepare_paired_prefix(ticket, checkpoint)
    }

    /// Commit a fully prepared memory pair using only ownership/trie mutation.
    pub(crate) fn commit_memory_paired_prefix(
        &mut self,
        prepared: PreparedPairedPrefix,
    ) -> Result<(), PairedCacheError> {
        self.memory.commit_prepared_pair(prepared)
    }

    #[cfg(test)]
    fn store_paired(
        &mut self,
        prefix_tokens: &[u32],
        cache: &AnyCache,
        snapshot: DFlashSnapshot,
        _checkpoint_id: Option<&str>,
    ) -> Result<(), PairedCacheError> {
        let ticket = self.paired_prepare_ticket();
        let _exec = higgs_models::mlx_exec::acquire();
        let prepared = PagedPrefixCache::prepare_paired_prefix_from_parts(
            ticket,
            prefix_tokens,
            cache,
            snapshot,
        )?;
        self.commit_memory_paired_prefix(prepared)
    }

    fn persist_target(
        &mut self,
        prefix_tokens: &[u32],
        cache: &AnyCache,
        checkpoint_id: Option<&str>,
    ) {
        if self.storage.is_none() || prefix_tokens.len() < self.min_tokens_to_persist {
            return;
        }

        let stored_len = prefix_tokens.len() / self.block_size * self.block_size;
        if stored_len < self.min_tokens_to_persist {
            return;
        }
        let Some(tokens_to_store) = prefix_tokens.get(..stored_len) else {
            return;
        };
        let token_hash = hash_tokens(tokens_to_store);

        let Some(storage) = self.storage.as_mut() else {
            return;
        };
        let header = storage.header().clone();
        let layers = match snapshot_layers(cache, &header, stored_len) {
            Ok(layers) => layers,
            Err(DiskCacheError::Unsupported(reason)) => {
                tracing::debug!(reason, "Skipping disk prefix cache store");
                return;
            }
            Err(error) => {
                tracing::warn!(error = %error, "Failed to build disk prefix cache snapshot");
                return;
            }
        };

        let token_session_id = hash_tokens_for_session(tokens_to_store);
        if let Err(error) = storage.save_blocks(token_session_id, token_hash, stored_len, &layers) {
            tracing::warn!(error = %error, "Failed to persist disk prefix cache snapshot");
        }
        if let Some(checkpoint) = checkpoint_id {
            let checkpoint_session_id = hash_checkpoint_id(checkpoint);
            if checkpoint_session_id != token_session_id {
                if let Err(error) =
                    storage.save_blocks(checkpoint_session_id, token_hash, stored_len, &layers)
                {
                    tracing::warn!(
                        checkpoint_id = checkpoint,
                        error = %error,
                        "Failed to persist named disk prefix cache checkpoint"
                    );
                }
            }
        }
    }

    pub const fn len(&self) -> usize {
        self.memory.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    pub const fn paired_entry_count(&self) -> usize {
        self.memory.paired_entry_count()
    }

    pub fn paired_stats(&self) -> PairedPrefixCacheStats {
        self.memory.paired_stats()
    }

    pub fn clear(&mut self) {
        self.memory.clear();
    }

    fn load_snapshot_match(&mut self, tokens: &[u32], session_id: u64) -> Option<PagedPrefixMatch> {
        let storage = self.storage.as_ref()?;
        let snapshot = match storage.load_blocks(session_id) {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => return None,
            Err(error) => {
                tracing::warn!(error = %error, "Failed to load disk prefix cache snapshot");
                return None;
            }
        };
        if snapshot.token_count > tokens.len() {
            return None;
        }
        let prefix_tokens = tokens.get(..snapshot.token_count)?;
        if hash_tokens(prefix_tokens) != snapshot.token_hash {
            tracing::debug!("Skipping disk prefix cache snapshot with mismatched token hash");
            return None;
        }
        let header = storage.header().clone();
        let cache = match materialize_snapshot(&snapshot, &header) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::warn!(error = %error, "Failed to materialize disk prefix cache snapshot");
                return None;
            }
        };
        self.memory
            .store_disk_refresh_preserving_pair(prefix_tokens, &cache);
        Some(PagedPrefixMatch {
            prefix_len: snapshot.token_count,
            cache,
        })
    }
}

pub fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn hash_tokens_for_session(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv_byte(hash, b't');
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    hash
}

#[derive(Debug, Clone, Copy)]
struct BlockPrefixHash {
    len: usize,
    token_hash: u64,
    session_id: u64,
}

fn block_prefix_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockPrefixHash> {
    if block_size == 0 {
        return Vec::new();
    }
    let max_len = tokens.len() / block_size * block_size;
    let mut prefixes = Vec::with_capacity(max_len / block_size);
    let mut token_hash = FNV_OFFSET;
    let mut session_hash = fnv_byte(FNV_OFFSET, b't');
    for (index, token) in tokens.iter().take(max_len).enumerate() {
        for byte in token.to_le_bytes() {
            token_hash = fnv_byte(token_hash, byte);
            session_hash = fnv_byte(session_hash, byte);
        }
        let len = index + 1;
        if len % block_size == 0 {
            prefixes.push(BlockPrefixHash {
                len,
                token_hash,
                session_id: session_hash,
            });
        }
    }
    prefixes
}

fn snapshot_metadata_matches(
    tokens: &[u32],
    metadata: DiskCacheEntryMetadata,
    min_prefix_len: usize,
) -> bool {
    if metadata.token_count < min_prefix_len || metadata.token_count > tokens.len() {
        return false;
    }
    let Some(prefix_tokens) = tokens.get(..metadata.token_count) else {
        return false;
    };
    hash_tokens(prefix_tokens) == metadata.token_hash
}

fn hash_checkpoint_id(checkpoint_id: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv_byte(hash, b'c');
    for byte in checkpoint_id.as_bytes() {
        hash = fnv_byte(hash, *byte);
    }
    hash
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

fn snapshot_layers(
    cache: &AnyCache,
    header: &DiskCacheFileHeader,
    stored_len: usize,
) -> Result<Vec<DiskCacheLayer>, DiskCacheError> {
    let AnyCache::KV(layers) = cache else {
        return Err(DiskCacheError::Unsupported("hybrid caches are memory-only"));
    };
    let block_count = stored_len / header.block_size;
    let block_size_i32 =
        i32::try_from(header.block_size).map_err(|_| DiskCacheError::Overflow("block_size"))?;
    let block_elems = header
        .block_size
        .checked_mul(header.num_kv_heads)
        .and_then(|value| value.checked_mul(header.head_dim))
        .ok_or(DiskCacheError::Overflow("block elements"))?;

    let mut disk_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let Some(kv) = layer.as_ref() else {
            disk_layers.push(DiskCacheLayer { blocks: Vec::new() });
            continue;
        };
        if kv.is_quantized() {
            return Err(DiskCacheError::Unsupported(
                "TurboQuant caches are memory-only",
            ));
        }
        let (Some(keys), Some(values)) = (kv.keys(), kv.values()) else {
            disk_layers.push(DiskCacheLayer { blocks: Vec::new() });
            continue;
        };
        validate_array_layout(keys, header, stored_len)?;
        validate_array_layout(values, header, stored_len)?;

        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            let start_usize = block_index
                .checked_mul(header.block_size)
                .ok_or(DiskCacheError::Overflow("block start"))?;
            let start =
                i32::try_from(start_usize).map_err(|_| DiskCacheError::Overflow("block start"))?;
            let end = start
                .checked_add(block_size_i32)
                .ok_or(DiskCacheError::Overflow("block end"))?;
            let k_block = array_block_to_f16(keys, start, end, block_elems)?;
            let v_block = array_block_to_f16(values, start, end, block_elems)?;
            blocks.push(DiskCacheBlock {
                k: k_block,
                v: v_block,
            });
        }
        disk_layers.push(DiskCacheLayer { blocks });
    }
    Ok(disk_layers)
}

fn validate_array_layout(
    array: &Array,
    header: &DiskCacheFileHeader,
    stored_len: usize,
) -> Result<(), DiskCacheError> {
    let shape = array.shape();
    let heads = shape_dim(shape, 1, "num_kv_heads")?;
    let tokens = shape_dim(shape, 2, "tokens")?;
    let head_dim = shape_dim(shape, 3, "head_dim")?;
    if heads != header.num_kv_heads || head_dim != header.head_dim || tokens < stored_len {
        return Err(DiskCacheError::Format(format!(
            "array layout mismatch: shape={shape:?}, expected heads={} head_dim={} tokens>={stored_len}",
            header.num_kv_heads, header.head_dim
        )));
    }
    Ok(())
}

fn shape_dim(shape: &[i32], index: usize, label: &'static str) -> Result<usize, DiskCacheError> {
    let value = shape
        .get(index)
        .copied()
        .ok_or_else(|| DiskCacheError::Format(format!("array missing {label} dimension")))?;
    usize::try_from(value).map_err(|_| DiskCacheError::Format(format!("invalid {label} dimension")))
}

fn array_block_to_f16(
    array: &Array,
    start: i32,
    end: i32,
    block_elems: usize,
) -> Result<Vec<f16>, DiskCacheError> {
    let block =
        slice_axis2(array, start, end).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let block_f16 = block
        .as_dtype(Dtype::Float16)
        .map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let data = block_f16.as_slice::<f16>().to_vec();
    if data.len() != block_elems {
        return Err(DiskCacheError::Format(
            "sliced block element count does not match layout".to_owned(),
        ));
    }
    Ok(data)
}

fn materialize_snapshot(
    snapshot: &DiskCacheSnapshot,
    header: &DiskCacheFileHeader,
) -> Result<AnyCache, DiskCacheError> {
    let layers: Result<Vec<_>, _> = snapshot
        .layers
        .iter()
        .map(|layer| {
            if layer.blocks.is_empty() {
                return Ok(Some(SteppingKeyValueCache::new()));
            }
            materialize_layer(layer, header).map(Some)
        })
        .collect();
    Ok(AnyCache::KV(layers?))
}

fn materialize_layer(
    layer: &DiskCacheLayer,
    header: &DiskCacheFileHeader,
) -> Result<SteppingKeyValueCache, DiskCacheError> {
    let shape = [
        1,
        i32::try_from(header.num_kv_heads).map_err(|_| DiskCacheError::Overflow("num_kv_heads"))?,
        i32::try_from(header.block_size).map_err(|_| DiskCacheError::Overflow("block_size"))?,
        i32::try_from(header.head_dim).map_err(|_| DiskCacheError::Overflow("head_dim"))?,
    ];
    let key_arrays: Vec<Array> = layer
        .blocks
        .iter()
        .map(|block| Array::from_slice(&block.k, &shape))
        .collect();
    let value_arrays: Vec<Array> = layer
        .blocks
        .iter()
        .map(|block| Array::from_slice(&block.v, &shape))
        .collect();
    let keys =
        concat_blocks(key_arrays).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    let values =
        concat_blocks(value_arrays).map_err(|error| DiskCacheError::Mlx(format!("{error}")))?;
    SteppingKeyValueCache::from_arrays(keys, values)
        .map_err(|error| DiskCacheError::Mlx(format!("{error}")))
}

fn concat_blocks(mut arrays: Vec<Array>) -> Result<Array, Exception> {
    if arrays.len() == 1 {
        return arrays
            .pop()
            .ok_or_else(|| Exception::custom("missing disk cache block"));
    }
    concatenate_axis(&arrays, 2)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use higgs_models::cache::KeyValueCache;
    use higgs_models::dflash::{DFlashConfig, DFlashDrafter, DFlashSnapshot};

    fn make_kv_cache_with_value(num_layers: usize, seq_len: i32, value: f32) -> AnyCache {
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| {
                let elem_count = usize::try_from(2 * seq_len * 4).unwrap();
                let values = vec![value; elem_count];
                let keys = Array::from_slice(&values, &[1, 2, seq_len, 4]);
                let vals = Array::from_slice(&values, &[1, 2, seq_len, 4]);
                Some(SteppingKeyValueCache::from_arrays(keys, vals).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    fn make_kv_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        make_kv_cache_with_value(num_layers, seq_len, 1.0)
    }

    fn kv_cache_has_value(cache: &AnyCache, value: f32) -> bool {
        let AnyCache::KV(layers) = cache else {
            return false;
        };
        let keys = layers[0].as_ref().unwrap().keys().unwrap();
        let element_count = keys
            .shape()
            .iter()
            .map(|dim| usize::try_from(*dim).unwrap())
            .product();
        let expected = Array::from_slice(&vec![value; element_count], keys.shape());
        keys.array_eq(&expected, None)
            .unwrap()
            .all(None)
            .unwrap()
            .item::<bool>()
    }

    fn dflash_snapshot(boundary: i32) -> DFlashSnapshot {
        let config: DFlashConfig = serde_json::from_str(
            r#"{
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8,
                "vocab_size": 8,
                "dflash_config": {
                    "target_layer_ids": [0]
                }
            }"#,
        )
        .unwrap();
        let mut drafter = DFlashDrafter::new(config).unwrap();
        let cache = drafter.make_cache();
        let taps = (boundary == 1)
            .then(|| Array::zeros::<f32>(&[1, 1, 4]).unwrap())
            .into_iter()
            .collect::<Vec<_>>();
        let _exec = higgs_models::mlx_exec::acquire();
        drafter.seal_after_taps(cache, &taps, boundary).unwrap()
    }

    fn find_memory_pair(
        cache: &mut DiskPrefixCache,
        tokens: &[u32],
    ) -> Option<crate::paged_prefix_cache::MaterializedPairedPrefix> {
        let plan = cache.plan_memory_paired_prefix(tokens).unwrap()?;
        let (matched, touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };
        assert!(
            cache.touch_memory_paired(touch),
            "successful paired materialization must refresh memory LRU"
        );
        Some(matched)
    }

    #[test]
    fn disk_cache_restores_into_empty_memory_cache() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 32,
        };
        let tokens: Vec<u32> = (0..64).collect();
        let cache = make_kv_cache(2, 64);

        let mut writer = DiskPrefixCache::new(8, 32, config.clone(), 2, 4).unwrap();
        writer.store(&tokens, &cache, Some("checkpoint-a"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 32, config, 2, 4).unwrap();
        let mut query = tokens.clone();
        query.push(999);
        let matched = reader.find_longest_prefix(&query, None).unwrap();
        assert_eq!(matched.prefix_len, 64);
        match matched.cache {
            AnyCache::KV(layers) => {
                assert_eq!(layers.len(), 2);
                let kv = layers[0].as_ref().unwrap();
                assert_eq!(KeyValueCache::offset(kv), 64);
                assert_eq!(kv.keys().unwrap().shape(), &[1, 2, 64, 4]);
            }
            AnyCache::Hybrid(_) => panic!("expected KV cache"),
        }
    }

    #[test]
    fn paired_store_is_available_from_same_process_memory() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 1,
        };
        let tokens = vec![7];
        let cache = make_kv_cache(1, 1);
        let mut prefix_cache = DiskPrefixCache::new(8, 1, config, 2, 4).unwrap();

        prefix_cache
            .store_paired(&tokens, &cache, dflash_snapshot(1), Some("paired"))
            .unwrap();

        let matched = find_memory_pair(&mut prefix_cache, &tokens)
            .expect("same-process paired state should remain in memory");
        assert_eq!(matched.prefix_len, 1);
        matched.cache.validate_absolute_boundary(1).unwrap();
        assert_eq!(matched.dflash_cache.position(), 1);
    }

    #[test]
    fn paired_memory_selection_returns_an_owned_post_lock_plan() {
        let mut cache = DiskPrefixCache::memory_only(4, 1);
        let tokens = vec![7];
        let target = make_kv_cache(1, 1);
        let snapshot = dflash_snapshot(1);
        let ticket = cache.paired_prepare_ticket();
        let prepared = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(ticket, &tokens, &target, snapshot)
                .unwrap()
        };
        assert!(!higgs_models::mlx_exec::held());
        cache.commit_memory_paired_prefix(prepared).unwrap();
        let mut query = tokens;
        query.push(8);

        assert!(!higgs_models::mlx_exec::held());
        let plan = cache
            .plan_memory_paired_prefix(&query)
            .unwrap()
            .expect("memory pair should be selected without MLX work");

        cache.memory.clear();
        let _exec = higgs_models::mlx_exec::acquire();
        let (matched, _touch) = plan.materialize_unproven_for_test().unwrap();
        assert_eq!(matched.prefix_len, 1);
        assert_eq!(matched.dflash_cache.position(), 1);
    }

    #[test]
    fn disk_wrapper_exposes_and_clears_paired_memory_stats() {
        let mut cache = DiskPrefixCache::memory_only(4, 1);
        let tokens = vec![7];
        cache
            .store_paired(&tokens, &make_kv_cache(1, 1), dflash_snapshot(1), None)
            .unwrap();

        let stats = cache.paired_stats();
        assert_eq!(cache.paired_entry_count(), 1);
        assert_eq!(stats.entries, 1);
        assert!(stats.target_bytes > 0);
        assert!(stats.dflash_bytes > 0);

        cache.clear();
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
    }

    #[test]
    fn disk_wrapper_ttl_evicts_only_whole_memory_pairs() {
        let mut cache = DiskPrefixCache::memory_only(4, 1);
        let paired = vec![7];
        let target_only = vec![9];
        cache
            .store_paired(&paired, &make_kv_cache(1, 1), dflash_snapshot(1), None)
            .unwrap();
        cache.store(&target_only, &make_kv_cache(1, 1), None);
        cache.set_paired_idle_ttl(Some(Duration::ZERO));

        assert_eq!(cache.evict_idle_paired(), 1);
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
        assert!(
            find_memory_pair(&mut cache, &[7, 99]).is_none(),
            "TTL must remove the dSpark half"
        );
        assert!(
            cache.find_memory_prefix(&[7, 99]).is_none(),
            "TTL must remove the paired target half too"
        );
        assert!(
            cache.find_memory_prefix(&[9, 99]).is_some(),
            "unrelated target-only memory must retain its existing policy"
        );
    }

    #[test]
    fn paired_state_is_memory_only_while_ordinary_target_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 1,
        };
        let tokens = vec![7];
        let cache = make_kv_cache(1, 1);

        let mut writer = DiskPrefixCache::new(8, 1, config.clone(), 2, 4).unwrap();
        writer
            .store_paired(&tokens, &cache, dflash_snapshot(1), Some("paired"))
            .unwrap();
        writer.store(&tokens, &cache, Some("paired"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 1, config, 2, 4).unwrap();
        let mut query = tokens.clone();
        query.push(99);
        assert!(
            find_memory_pair(&mut reader, &query).is_none(),
            "dFlash state must never be synthesized from target-only disk data"
        );

        let target = reader
            .find_longest_prefix(&query, Some("paired"))
            .expect("ordinary target lookup should restore the persisted snapshot");
        assert_eq!(target.prefix_len, 1);
        target.cache.validate_absolute_boundary(1).unwrap();
        assert!(
            find_memory_pair(&mut reader, &query).is_none(),
            "loading target state from disk must remain explicitly target-only"
        );
    }

    #[test]
    fn deeper_disk_target_never_replaces_or_combines_with_shorter_memory_pair() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 1,
        };
        let paired_tokens = vec![7];
        let disk_tokens = vec![7, 8];
        let disk_target = make_kv_cache_with_value(1, 2, 2.0);

        let mut writer = DiskPrefixCache::new(8, 1, config.clone(), 2, 4).unwrap();
        writer.store(&disk_tokens, &disk_target, Some("deeper-target"));
        drop(writer);

        let paired_target = make_kv_cache_with_value(1, 1, 7.0);
        let mut reader = DiskPrefixCache::new(8, 1, config, 2, 4).unwrap();
        reader
            .store_paired(&paired_tokens, &paired_target, dflash_snapshot(1), None)
            .unwrap();
        let paired_before = reader.paired_stats();
        let query = vec![7, 8, 9];

        let paired = find_memory_pair(&mut reader, &query)
            .expect("paired lookup must select the exact in-memory capability");
        assert_eq!(paired.prefix_len, 1);
        assert!(kv_cache_has_value(&paired.cache, 7.0));
        assert_eq!(paired.dflash_cache.position(), 1);

        let target = reader
            .find_longest_prefix(&query, Some("deeper-target"))
            .expect("ordinary target lookup may select the deeper disk snapshot");
        assert_eq!(target.prefix_len, 2);
        assert!(kv_cache_has_value(&target.cache, 2.0));

        let paired_after = find_memory_pair(&mut reader, &query)
            .expect("loading a deeper target-only disk endpoint must preserve the shorter pair");
        assert_eq!(paired_after.prefix_len, 1);
        assert!(kv_cache_has_value(&paired_after.cache, 7.0));
        assert_eq!(paired_after.dflash_cache.position(), 1);
        assert_eq!(
            reader.paired_stats(),
            paired_before,
            "a deeper disk refresh must not mutate paired endpoint accounting"
        );
    }

    #[test]
    fn target_refresh_demotes_pair_even_when_disk_persistence_fails() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 0,
            min_tokens_to_persist: 1,
        };
        let tokens = vec![7];
        let cache = make_kv_cache(1, 1);
        let mut prefix_cache = DiskPrefixCache::new(8, 1, config, 2, 4).unwrap();

        prefix_cache
            .store_paired(&tokens, &cache, dflash_snapshot(1), Some("paired"))
            .expect("memory-only paired publication should succeed");
        prefix_cache.store(&tokens, &cache, Some("paired"));

        assert!(
            find_memory_pair(&mut prefix_cache, &tokens).is_none(),
            "a target-only refresh must demote speculative continuity regardless of disk outcome"
        );
        assert!(
            prefix_cache
                .find_disk_prefix_candidate(&tokens, Some("paired"), 0)
                .is_none(),
            "the failed disk write must not publish target metadata"
        );
    }

    #[test]
    fn same_key_disk_refresh_preserves_the_original_whole_memory_pair() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 1,
        };
        let tokens = vec![7];
        let disk_target = make_kv_cache_with_value(1, 1, 2.0);
        let paired_target = make_kv_cache_with_value(1, 1, 7.0);
        let mut prefix_cache = DiskPrefixCache::new(8, 1, config, 2, 4).unwrap();
        prefix_cache.store(&tokens, &disk_target, Some("paired"));
        prefix_cache
            .store_paired(&tokens, &paired_target, dflash_snapshot(1), Some("paired"))
            .unwrap();

        let candidate = prefix_cache
            .find_disk_prefix_candidate(&tokens, Some("paired"), 0)
            .expect("ordinary target store should persist its target state");
        let loaded = prefix_cache
            .load_disk_prefix_candidate(&tokens, candidate)
            .expect("forced target-only refresh should load");
        assert!(
            kv_cache_has_value(&loaded.cache, 2.0),
            "the current ordinary request should receive the materialized disk target"
        );

        let retained = find_memory_pair(&mut prefix_cache, &tokens)
            .expect("a trusted exact-key disk refresh must leave the pair selectable");
        assert!(
            kv_cache_has_value(&retained.cache, 7.0),
            "disk refresh must leave the original paired target half untouched"
        );
        assert_eq!(retained.dflash_cache.position(), 1);
    }

    #[test]
    fn disk_prefix_candidates_respect_min_prefix_len() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 32,
        };
        let tokens: Vec<u32> = (0..64).collect();
        let cache = make_kv_cache(1, 64);

        let mut writer = DiskPrefixCache::new(8, 32, config.clone(), 2, 4).unwrap();
        writer.store(&tokens, &cache, Some("checkpoint-a"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 32, config, 2, 4).unwrap();
        let mut query = tokens.clone();
        query.push(999);

        assert!(
            reader
                .find_disk_prefix_candidate(&query, None, 65)
                .is_none(),
            "disk should not materialize a snapshot that cannot beat memory"
        );

        let candidate = reader
            .find_disk_prefix_candidate(&query, None, 1)
            .expect("stored token prefix should be a disk candidate");
        assert_eq!(candidate.prefix_len(), 64);
        let matched = reader
            .load_disk_prefix_candidate(&query, candidate)
            .expect("candidate should materialize under the MLX gate");
        assert_eq!(matched.prefix_len, 64);
    }

    #[test]
    fn named_checkpoint_validates_prompt_hash() {
        let dir = tempfile::tempdir().unwrap();
        let config = DiskPrefixCacheConfig {
            disk_path: dir.path().join("prefix.bin"),
            max_disk_blocks: 16,
            min_tokens_to_persist: 32,
        };
        let tokens: Vec<u32> = (0..64).collect();
        let cache = make_kv_cache(1, 64);

        let mut writer = DiskPrefixCache::new(8, 32, config.clone(), 2, 4).unwrap();
        writer.store(&tokens, &cache, Some("checkpoint-a"));
        drop(writer);

        let mut reader = DiskPrefixCache::new(8, 32, config, 2, 4).unwrap();
        let wrong_tokens: Vec<u32> = (1000..1064).collect();
        assert!(
            reader
                .find_longest_prefix(&wrong_tokens, Some("checkpoint-a"))
                .is_none()
        );
    }

    #[test]
    fn block_prefix_hashes_match_legacy_hashes() {
        let tokens: Vec<u32> = (0..96).collect();
        let prefixes = block_prefix_hashes(&tokens, 32);
        assert_eq!(prefixes.len(), 3);
        for prefix in prefixes {
            let prefix_tokens = &tokens[..prefix.len];
            assert_eq!(prefix.token_hash, hash_tokens(prefix_tokens));
            assert_eq!(prefix.session_id, hash_tokens_for_session(prefix_tokens));
        }
    }
}
