#![allow(
    clippy::all,
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::format_collect,
    clippy::impl_trait_in_params,
    clippy::indexing_slicing,
    clippy::let_and_return,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::panic,
    clippy::ref_option,
    clippy::semicolon_if_nothing_returned,
    clippy::shadow_reuse,
    clippy::unnecessary_semicolon,
    clippy::unnecessary_wraps,
    clippy::unwrap_used
)]

//! Durable, model-bound prefix-cache entries.
//!
//! Files are atomically written below `<dir>/<first-two-key-hex>/<key>.hkv`.
//! File mtime is the v1 frecency signal: hits touch files and stores evict the
//! least recently touched entries until the configured budget is respected.

use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};
use higgs_models::{AnyCache, LayerCache, cache::SteppingKeyValueCache};
use mlx_rs::{Array, Dtype, complex64};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"HKV1";
const VERSION: u32 = 2;
const CACHE_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreIdentity {
    pub model_id: String,
    pub quant: String,
    pub config_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub chat_template_hash: [u8; 32],
    /// Fast heuristic fingerprint over the model directory's weight files.
    ///
    /// This is intentionally NOT a content hash: hashing multi-gigabyte
    /// safetensors files on every startup would be unacceptably slow. It
    /// digests each `*.safetensors` file's (name, byte size, mtime seconds)
    /// tuple instead, which is enough to detect weights swapped in place
    /// (e.g. a re-quantization or re-download) without reading file bodies.
    pub weights_hash: [u8; 32],
}

impl StoreIdentity {
    pub fn from_model_dir(model_dir: &Path, quant: String) -> Self {
        let hash_file = |name: &str| {
            fs::read(model_dir.join(name)).map_or_else(|_| sha256(&[]), |b| sha256(&b))
        };
        let tokenizer_hash = ["tokenizer.json", "tokenizer.model"]
            .iter()
            .find_map(|name| fs::read(model_dir.join(name)).ok().map(|b| sha256(&b)))
            .unwrap_or_else(|| sha256(&[]));
        let chat_template_hash = fs::read(model_dir.join("chat_template.jinja"))
            .map_or_else(|_| sha256(&[]), |b| sha256(&b));
        Self {
            model_id: model_dir.to_string_lossy().into_owned(),
            quant,
            config_hash: hash_file("config.json"),
            tokenizer_hash,
            chat_template_hash,
            weights_hash: weights_fingerprint(model_dir),
        }
    }

    #[cfg(test)]
    fn for_tests() -> Self {
        Self {
            model_id: "test-model".into(),
            quant: "dense".into(),
            config_hash: [1; 32],
            tokenizer_hash: [2; 32],
            chat_template_hash: [3; 32],
            weights_hash: [4; 32],
        }
    }
}

/// Heuristic weight fingerprint: digests the sorted (name, size, mtime)
/// tuples of `*.safetensors` files directly inside `model_dir`. See
/// `StoreIdentity::weights_hash` for why this isn't a content hash.
fn weights_fingerprint(model_dir: &Path) -> [u8; 32] {
    let mut entries: Vec<(String, u64, u64)> = fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            let mtime_secs = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            Some((
                entry.file_name().to_string_lossy().into_owned(),
                meta.len(),
                mtime_secs,
            ))
        })
        .collect();
    entries.sort();
    let mut hasher = Sha256::new();
    for (name, len, mtime) in entries {
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
        hasher.update(len.to_le_bytes());
        hasher.update(mtime.to_le_bytes());
        hasher.update(b"\n");
    }
    hasher.finalize().into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DensePayload(pub Vec<u8>);
impl DensePayload {
    #[cfg(test)]
    fn test_payload() -> Self {
        Self(vec![1, 2, 3, 4])
    }
    #[cfg(test)]
    fn sized(n: usize) -> Self {
        Self(vec![7; n])
    }
}

pub struct DiskPrefixStore {
    dir: PathBuf,
    budget: u64,
    identity: StoreIdentity,
}

impl DiskPrefixStore {
    pub fn new(
        dir: impl AsRef<Path>,
        budget: u64,
        identity: StoreIdentity,
    ) -> std::io::Result<Self> {
        fs::create_dir_all(dir.as_ref())?;
        let store = Self {
            dir: dir.as_ref().to_path_buf(),
            budget,
            identity,
        };
        store.cleanup_temps()?;
        Ok(store)
    }
    pub fn store_payload(
        &self,
        tokens: &[u32],
        block_size: u32,
        payload: &DensePayload,
    ) -> std::io::Result<()> {
        let token_count = u32::try_from(tokens.len())
            .map_err(|_| std::io::Error::other("token count overflow"))?;
        let key = key(tokens, block_size as usize);
        let path = self.path_for(&key);
        let parent = path
            .parent()
            .ok_or_else(|| std::io::Error::other("missing store parent"))?;
        fs::create_dir_all(parent)?;
        let tmp = unique_tmp_path(&path);
        let mut file = File::create(&tmp)?;
        file.write_all(MAGIC)?;
        write_u32(&mut file, VERSION)?;
        write_string(&mut file, &self.identity.model_id)?;
        write_string(&mut file, &self.identity.quant)?;
        file.write_all(&self.identity.config_hash)?;
        file.write_all(&self.identity.tokenizer_hash)?;
        file.write_all(&self.identity.chat_template_hash)?;
        file.write_all(&self.identity.weights_hash)?;
        write_u32(&mut file, block_size)?;
        write_u32(&mut file, token_count)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 1)?;
        write_u64(
            &mut file,
            u64::try_from(payload.0.len())
                .map_err(|_| std::io::Error::other("payload too large"))?,
        )?;
        file.write_all(&payload.0)?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)?;
        self.evict()?;
        Ok(())
    }
    pub fn load_payload(&self, tokens: &[u32]) -> Option<DensePayload> {
        for (path, _) in self.files().ok()? {
            if let Some(payload) = self.read_payload(&path, tokens) {
                let _ = touch(&path);
                return Some(payload);
            }
        }
        None
    }

    /// Persist a block-aligned KV cache. Unsupported recurrent/hybrid cache
    /// layers are rejected: serving a partial cache would be incorrect.
    pub fn store_cache(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache: &AnyCache,
    ) -> std::io::Result<()> {
        let block_size_usize = usize::try_from(block_size)
            .map_err(|_| std::io::Error::other("block size overflow"))?;
        if block_size_usize == 0 {
            return Err(std::io::Error::other("block size must be positive"));
        }
        let token_count = u32::try_from(tokens.len() / block_size_usize * block_size_usize)
            .map_err(|_| std::io::Error::other("token count overflow"))?;
        let layers = cache_layers(cache)?;

        // Serialize the layer section into memory first so its checksum can
        // be recorded in the header before any of it is written to disk.
        let mut payload = Vec::new();
        write_u32(
            &mut payload,
            u32::try_from(layers.len()).map_err(|_| std::io::Error::other("too many layers"))?,
        )?;
        for layer in layers {
            match layer {
                DiskLayer::Empty => payload.write_all(&[0])?,
                DiskLayer::Dense(keys, values) => {
                    payload.write_all(&[1])?;
                    write_array(&mut payload, keys)?;
                    write_array(&mut payload, values)?;
                }
                DiskLayer::Turbo(arrays) => {
                    payload.write_all(&[2])?;
                    for array in arrays {
                        write_array(&mut payload, array)?;
                    }
                }
            }
        }
        let payload_checksum = sha256(&payload);
        let payload_len =
            u64::try_from(payload.len()).map_err(|_| std::io::Error::other("payload too large"))?;

        let key = key(tokens, block_size_usize);
        let path = self.path_for(&key);
        let parent = path
            .parent()
            .ok_or_else(|| std::io::Error::other("missing store parent"))?;
        fs::create_dir_all(parent)?;
        let tmp = unique_tmp_path(&path);
        let mut file = File::create(&tmp)?;
        write_header(
            &mut file,
            CACHE_VERSION,
            &self.identity,
            block_size,
            token_count,
            &payload_checksum,
            payload_len,
        )?;
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)?;
        self.evict()
    }

    /// Load a compatible cache. `prototype` supplies cache configuration and
    /// TurboQuant contexts, which are model-bound and intentionally never read
    /// from disk.
    ///
    /// Selects the longest stored prefix of `tokens` deterministically:
    /// candidate block-aligned prefix lengths are probed longest-first by
    /// their content-addressed key, rather than scanning the store in
    /// filesystem order.
    pub fn load_cache(
        &self,
        tokens: &[u32],
        block_size: u32,
        prototype: &AnyCache,
    ) -> std::io::Result<Option<(usize, AnyCache)>> {
        let block_size_usize = usize::try_from(block_size)
            .map_err(|_| std::io::Error::other("block size overflow"))?;
        if block_size_usize == 0 {
            return Ok(None);
        }
        let max_len = tokens.len() / block_size_usize * block_size_usize;
        let mut candidate_len = max_len;
        while candidate_len >= block_size_usize {
            let key = key(&tokens[..candidate_len], block_size_usize);
            let path = self.path_for(&key);
            if path.exists() {
                match self.read_cache(&path, tokens, prototype) {
                    Ok(Some((prefix_len, cache))) => {
                        let _ = touch(&path);
                        return Ok(Some((prefix_len, cache)));
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::debug!(path = %path.display(), %error, "disk prefix cache entry ignored");
                    }
                }
            }
            candidate_len -= block_size_usize;
        }
        Ok(None)
    }

    fn read_cache(
        &self,
        path: &Path,
        tokens: &[u32],
        prototype: &AnyCache,
    ) -> std::io::Result<Option<(usize, AnyCache)>> {
        let mut file = File::open(path)?;
        let Some((block_size, token_count, payload_checksum, payload_len)) =
            read_header(&mut file, CACHE_VERSION, &self.identity)?
        else {
            return Ok(None);
        };
        let block_size = usize::try_from(block_size)
            .map_err(|_| std::io::Error::other("block size overflow"))?;
        let token_count = usize::try_from(token_count)
            .map_err(|_| std::io::Error::other("token count overflow"))?;
        if !Self::matches_request(path, tokens, block_size, token_count) {
            return Ok(None);
        }
        // Bound the allocation by what the file could actually contain: a
        // corrupted or adversarial payload_len must become a clean miss,
        // never an OOM/capacity-overflow panic from an oversized `vec![]`.
        let file_len = file.metadata()?.len();
        let remaining = file_len.saturating_sub(file.stream_position()?);
        if payload_len > remaining {
            tracing::debug!(
                payload_len,
                remaining,
                reason = "payload length exceeds remaining file size",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        let payload_len_usize = usize::try_from(payload_len)
            .map_err(|_| std::io::Error::other("payload length overflow"))?;
        let mut payload = vec![0_u8; payload_len_usize];
        file.read_exact(&mut payload)?;
        if sha256(&payload) != payload_checksum {
            tracing::debug!(
                reason = "tensor payload checksum mismatch",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        // Checksum verified: safe to materialize arrays from `payload` below.
        let mut cursor = Cursor::new(payload);
        let count = usize::try_from(
            read_u32(&mut cursor).ok_or_else(|| std::io::Error::other("missing layer count"))?,
        )
        .map_err(|_| std::io::Error::other("layer count overflow"))?;
        let configs = prototype_layers(prototype)?;
        if count != configs.len() {
            tracing::debug!(
                stored_layers = count,
                prototype_layers = configs.len(),
                reason = "cache layer count mismatch",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        let Some(layers) = parse_cache_layers(&mut cursor, configs, token_count)? else {
            return Ok(None);
        };
        if cursor.position() != payload_len {
            tracing::debug!(
                reason = "trailing tensor payload data",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        if file.read(&mut [0; 1])? != 0 {
            tracing::debug!(
                reason = "trailing cache data",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        match prototype {
            AnyCache::KV(_) => Ok(Some((
                token_count,
                AnyCache::KV(
                    layers
                        .into_iter()
                        .map(|layer| match layer {
                            Some(LayerCache::KV(kv)) => Some(kv),
                            _ => None,
                        })
                        .collect(),
                ),
            ))),
            AnyCache::Hybrid(_) => Ok(Some((token_count, AnyCache::Hybrid(layers)))),
        }
    }

    fn matches_request(path: &Path, tokens: &[u32], block_size: usize, token_count: usize) -> bool {
        if block_size == 0 {
            tracing::debug!(
                reason = "zero block size",
                "Disk prefix cache entry rejected"
            );
            return false;
        }
        if token_count > tokens.len() {
            tracing::debug!(
                token_count,
                prompt_tokens = tokens.len(),
                reason = "stored prefix is longer than request",
                "Disk prefix cache entry rejected"
            );
            return false;
        }
        if token_count % block_size != 0 {
            tracing::debug!(
                token_count,
                block_size,
                reason = "stored token count is not block-aligned",
                "Disk prefix cache entry rejected"
            );
            return false;
        }
        if key(&tokens[..token_count], block_size)
            != path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
        {
            tracing::debug!(
                reason = "prompt token hash mismatch",
                "Disk prefix cache entry rejected"
            );
            return false;
        }
        true
    }

    fn read_payload(&self, path: &Path, tokens: &[u32]) -> Option<DensePayload> {
        let mut file = File::open(path).ok()?;
        let mut magic = [0; 4];
        file.read_exact(&mut magic).ok()?;
        if &magic != MAGIC
            || read_u32(&mut file)? != VERSION
            || read_string(&mut file)? != self.identity.model_id
            || read_string(&mut file)? != self.identity.quant
        {
            return None;
        }
        for expected in [
            &self.identity.config_hash,
            &self.identity.tokenizer_hash,
            &self.identity.chat_template_hash,
            &self.identity.weights_hash,
        ] {
            let mut actual = [0; 32];
            file.read_exact(&mut actual).ok()?;
            if actual != *expected {
                return None;
            }
        }
        let block_size = usize::try_from(read_u32(&mut file)?).ok()?;
        let token_count = usize::try_from(read_u32(&mut file)?).ok()?;
        let _layers = read_u32(&mut file)?;
        if read_u32(&mut file)? != 1
            || block_size == 0
            || token_count > tokens.len()
            || token_count % block_size != 0
            || key(&tokens[..token_count], block_size) != path.file_stem()?.to_str()?
        {
            return None;
        }
        let len_u64 = read_u64(&mut file)?;
        if len_u64 > self.budget {
            return None;
        }
        let len = usize::try_from(len_u64).ok()?;
        let mut payload = vec![0; len];
        file.read_exact(&mut payload).ok()?;
        let mut extra = [0; 1];
        if file.read(&mut extra).ok()? != 0 {
            return None;
        }
        Some(DensePayload(payload))
    }
    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(&key[..2]).join(format!("{key}.hkv"))
    }
    fn evict(&self) -> std::io::Result<()> {
        let mut files = self.files()?;
        let mut total: u64 = files.iter().map(|(_, m)| m.len()).sum();
        files.sort_by_key(|(_, m)| m.modified().ok());
        for (path, meta) in files {
            if total <= self.budget {
                break;
            }
            fs::remove_file(path)?;
            total = total.saturating_sub(meta.len());
        }
        Ok(())
    }
    fn files(&self) -> std::io::Result<Vec<(PathBuf, fs::Metadata)>> {
        let mut files = Vec::new();
        for dir in fs::read_dir(&self.dir)? {
            let dir = dir?;
            if !dir.file_type()?.is_dir() {
                continue;
            }
            for entry in fs::read_dir(dir.path())? {
                let entry = entry?;
                if entry.path().extension().and_then(|e| e.to_str()) == Some("hkv") {
                    files.push((entry.path(), entry.metadata()?));
                }
            }
        }
        Ok(files)
    }
    pub fn cleanup_temps(&self) -> std::io::Result<()> {
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                for child in fs::read_dir(entry.path())? {
                    let child = child?;
                    if is_tmp_file(&child.path()) {
                        fs::remove_file(child.path())?;
                    }
                }
            } else if is_tmp_file(&entry.path()) {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn corrupt_for_test(&self, tokens: &[u32], kind: Corruption) -> std::io::Result<()> {
        let path = self
            .files()?
            .into_iter()
            .find(|(p, _)| self.read_payload(p, tokens).is_some())
            .map(|(p, _)| p)
            .ok_or_else(|| std::io::Error::other("entry missing"))?;
        let mut bytes = fs::read(&path)?;
        match kind {
            Corruption::Truncate => {
                bytes.truncate(8);
            }
            Corruption::Magic => bytes[0] ^= 1,
            Corruption::Model => bytes[12] ^= 1,
            Corruption::Quant => bytes[25] ^= 1,
            Corruption::Tokenizer => bytes[40] ^= 1,
            // First byte of `weights_hash`, which follows
            // magic(4) + version(4) + model_id(4+10) + quant(4+5) +
            // config_hash(32) + tokenizer_hash(32) + chat_template_hash(32).
            Corruption::Weights => bytes[127] ^= 1,
        };
        fs::write(path, bytes)
    }
}

#[cfg(test)]
enum Corruption {
    Truncate,
    Model,
    Quant,
    Tokenizer,
    Magic,
    Weights,
}

enum DiskLayer<'a> {
    Empty,
    Dense(&'a Array, &'a Array),
    Turbo([&'a Array; 5]),
}

enum PrototypeLayer {
    Empty,
    Dense(higgs_models::turboquant::KvCacheConfig),
    Turbo(std::sync::Arc<higgs_models::turboquant::TurboQuantContext>),
}

fn cache_layers(cache: &AnyCache) -> std::io::Result<Vec<DiskLayer<'_>>> {
    let layers = match cache {
        AnyCache::KV(layers) => layers
            .iter()
            .map(|layer| match layer {
                None => Ok(DiskLayer::Empty),
                Some(kv) => cache_kv_layer(kv),
            })
            .collect(),
        AnyCache::Hybrid(layers) => layers
            .iter()
            .map(|layer| match layer {
                None => Ok(DiskLayer::Empty),
                Some(LayerCache::KV(kv)) => cache_kv_layer(kv),
                Some(LayerCache::Arrays(_)) => Err(std::io::Error::other(
                    "recurrent hybrid layer cannot be persisted",
                )),
            })
            .collect(),
    };
    layers
}

fn cache_kv_layer(kv: &SteppingKeyValueCache) -> std::io::Result<DiskLayer<'_>> {
    if kv.is_mla() {
        tracing::debug!("MLA latent prefixes not yet disk-persistable");
        return Err(std::io::Error::other(
            "MLA latent layer cannot be persisted",
        ));
    }
    if let Some((_, key_codes, key_norms, key_gammas, value_codes, value_norms)) = kv.turbo_arrays()
    {
        return Ok(DiskLayer::Turbo([
            key_codes,
            key_norms,
            key_gammas,
            value_codes,
            value_norms,
        ]));
    }
    match (kv.keys(), kv.values()) {
        (Some(keys), Some(values)) => Ok(DiskLayer::Dense(keys, values)),
        (None, None) => Ok(DiskLayer::Empty),
        _ => Err(std::io::Error::other("incomplete KV layer")),
    }
}

fn prototype_layers(cache: &AnyCache) -> std::io::Result<Vec<PrototypeLayer>> {
    let layers = match cache {
        AnyCache::KV(layers) => layers.iter().map(prototype_kv_layer).collect(),
        AnyCache::Hybrid(layers) => layers
            .iter()
            .map(|layer| match layer {
                None => Ok(PrototypeLayer::Empty),
                Some(LayerCache::KV(kv)) => prototype_kv(kv),
                Some(LayerCache::Arrays(_)) => Err(std::io::Error::other(
                    "recurrent hybrid layer cannot be restored",
                )),
            })
            .collect(),
    };
    layers
}

fn prototype_kv_layer(layer: &Option<SteppingKeyValueCache>) -> std::io::Result<PrototypeLayer> {
    let Some(kv) = layer else {
        return Ok(PrototypeLayer::Empty);
    };
    prototype_kv(kv)
}
fn prototype_kv(kv: &SteppingKeyValueCache) -> std::io::Result<PrototypeLayer> {
    if kv.is_mla() {
        tracing::debug!("MLA latent prefixes not yet disk-persistable");
        return Err(std::io::Error::other("MLA latent layer cannot be restored"));
    }
    if let Some(context) = kv.turbo_context() {
        return Ok(PrototypeLayer::Turbo(std::sync::Arc::clone(context)));
    }
    Ok(PrototypeLayer::Dense(kv.kv_cache_config()))
}

fn mlx_error(error: mlx_rs::error::Exception) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

/// Parse the layer section out of a checksum-verified payload cursor.
/// Returns `Ok(None)` for a recognized-but-mismatched layout (clean miss,
/// e.g. after a format change), and `Err` for I/O or array decode failures.
fn parse_cache_layers(
    cursor: &mut Cursor<Vec<u8>>,
    configs: Vec<PrototypeLayer>,
    token_count: usize,
) -> std::io::Result<Option<Vec<Option<LayerCache>>>> {
    let mut layers = Vec::with_capacity(configs.len());
    for config in configs {
        let mut tag = [0; 1];
        cursor.read_exact(&mut tag)?;
        layers.push(match (tag[0], config) {
            (0, PrototypeLayer::Empty) => None,
            (1, PrototypeLayer::Dense(config)) => Some(LayerCache::KV(
                SteppingKeyValueCache::from_arrays_with_config(
                    read_array(cursor)?,
                    read_array(cursor)?,
                    config,
                )
                .map_err(mlx_error)?,
            )),
            (2, PrototypeLayer::Turbo(context)) => Some(LayerCache::KV(
                SteppingKeyValueCache::from_turbo_arrays(
                    context,
                    read_array(cursor)?,
                    read_array(cursor)?,
                    read_array(cursor)?,
                    read_array(cursor)?,
                    read_array(cursor)?,
                    i32::try_from(token_count)
                        .map_err(|_| std::io::Error::other("token count overflow"))?,
                )
                .map_err(mlx_error)?,
            )),
            _ => {
                tracing::debug!(
                    tag = tag[0],
                    reason = "cache layer layout mismatch",
                    "Disk prefix cache entry rejected"
                );
                return Ok(None);
            }
        });
    }
    Ok(Some(layers))
}

fn write_header<W: Write>(
    w: &mut W,
    version: u32,
    identity: &StoreIdentity,
    block_size: u32,
    token_count: u32,
    payload_checksum: &[u8; 32],
    payload_len: u64,
) -> std::io::Result<()> {
    w.write_all(MAGIC)?;
    write_u32(w, version)?;
    write_string(w, &identity.model_id)?;
    write_string(w, &identity.quant)?;
    w.write_all(&identity.config_hash)?;
    w.write_all(&identity.tokenizer_hash)?;
    w.write_all(&identity.chat_template_hash)?;
    w.write_all(&identity.weights_hash)?;
    write_u32(w, block_size)?;
    write_u32(w, token_count)?;
    w.write_all(payload_checksum)?;
    write_u64(w, payload_len)
}

#[allow(clippy::type_complexity)]
fn read_header<R: Read>(
    r: &mut R,
    version: u32,
    identity: &StoreIdentity,
) -> std::io::Result<Option<(u32, u32, [u8; 32], u64)>> {
    let mut magic = [0; 4];
    if r.read_exact(&mut magic).is_err() {
        tracing::debug!(
            reason = "truncated header",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    if &magic != MAGIC {
        tracing::debug!(
            reason = "magic mismatch",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    if read_u32(r) != Some(version) {
        tracing::debug!(
            reason = "format version mismatch",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    if read_string(r).as_deref() != Some(&identity.model_id) {
        tracing::debug!(
            reason = "StoreIdentity model_id mismatch",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    if read_string(r).as_deref() != Some(&identity.quant) {
        tracing::debug!(
            reason = "StoreIdentity quant mismatch",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    for (expected, reason) in [
        (&identity.config_hash, "StoreIdentity config hash mismatch"),
        (
            &identity.tokenizer_hash,
            "StoreIdentity tokenizer hash mismatch",
        ),
        (
            &identity.chat_template_hash,
            "StoreIdentity chat template hash mismatch",
        ),
        (
            &identity.weights_hash,
            "StoreIdentity weights fingerprint mismatch",
        ),
    ] {
        let mut actual = [0; 32];
        if r.read_exact(&mut actual).is_err() {
            tracing::debug!(
                reason = "truncated StoreIdentity",
                "Disk prefix cache entry rejected"
            );
            return Ok(None);
        }
        if actual != *expected {
            tracing::debug!(%reason, "Disk prefix cache entry rejected");
            return Ok(None);
        }
    }
    let Some(block) = read_u32(r) else {
        return Ok(None);
    };
    let Some(tokens) = read_u32(r) else {
        return Ok(None);
    };
    let mut payload_checksum = [0; 32];
    if r.read_exact(&mut payload_checksum).is_err() {
        tracing::debug!(
            reason = "truncated payload checksum",
            "Disk prefix cache entry rejected"
        );
        return Ok(None);
    }
    let Some(payload_len) = read_u64(r) else {
        return Ok(None);
    };
    Ok(Some((block, tokens, payload_checksum, payload_len)))
}

fn write_array<W: Write>(w: &mut W, array: &Array) -> std::io::Result<()> {
    let dtype = dtype_tag(array.dtype())?;
    let shape = array.shape();
    w.write_all(&[
        dtype,
        u8::try_from(shape.len()).map_err(|_| std::io::Error::other("array rank overflow"))?,
    ])?;
    for &dim in shape {
        write_u32(
            w,
            u32::try_from(dim).map_err(|_| std::io::Error::other("negative array dimension"))?,
        )?;
    }
    let bytes = array_bytes(array)?;
    write_u64(
        w,
        u64::try_from(bytes.len()).map_err(|_| std::io::Error::other("array too large"))?,
    )?;
    w.write_all(&bytes)
}

fn read_array<R: Read>(r: &mut R) -> std::io::Result<Array> {
    let mut header = [0; 2];
    r.read_exact(&mut header)?;
    let dtype = tag_dtype(header[0])?;
    let rank = usize::from(header[1]);
    if rank > 8 {
        return Err(std::io::Error::other("array rank too large"));
    }
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(
            i32::try_from(read_u32(r).ok_or_else(|| std::io::Error::other("missing shape"))?)
                .map_err(|_| std::io::Error::other("shape overflow"))?,
        );
    }
    let count = shape
        .iter()
        .try_fold(1_usize, |n, &d| {
            usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
        })
        .ok_or_else(|| std::io::Error::other("array shape overflow"))?;
    let byte_len =
        usize::try_from(read_u64(r).ok_or_else(|| std::io::Error::other("missing byte length"))?)
            .map_err(|_| std::io::Error::other("byte length overflow"))?;
    if byte_len
        != count
            .checked_mul(dtype_size(dtype))
            .ok_or_else(|| std::io::Error::other("array too large"))?
    {
        return Err(std::io::Error::other("array byte length mismatch"));
    }
    let mut bytes = vec![0; byte_len];
    r.read_exact(&mut bytes)?;
    array_from_bytes(dtype, &shape, &bytes)
}

fn dtype_tag(dtype: Dtype) -> std::io::Result<u8> {
    match dtype {
        Dtype::Bool => Ok(1),
        Dtype::Uint8 => Ok(2),
        Dtype::Uint16 => Ok(3),
        Dtype::Uint32 => Ok(4),
        Dtype::Uint64 => Ok(5),
        Dtype::Int8 => Ok(6),
        Dtype::Int16 => Ok(7),
        Dtype::Int32 => Ok(8),
        Dtype::Int64 => Ok(9),
        Dtype::Float16 => Ok(10),
        Dtype::Float32 => Ok(11),
        Dtype::Float64 => Ok(12),
        Dtype::Bfloat16 => Ok(13),
        Dtype::Complex64 => Ok(14),
    }
}
fn tag_dtype(tag: u8) -> std::io::Result<Dtype> {
    match tag {
        1 => Ok(Dtype::Bool),
        2 => Ok(Dtype::Uint8),
        3 => Ok(Dtype::Uint16),
        4 => Ok(Dtype::Uint32),
        5 => Ok(Dtype::Uint64),
        6 => Ok(Dtype::Int8),
        7 => Ok(Dtype::Int16),
        8 => Ok(Dtype::Int32),
        9 => Ok(Dtype::Int64),
        10 => Ok(Dtype::Float16),
        11 => Ok(Dtype::Float32),
        12 => Ok(Dtype::Float64),
        13 => Ok(Dtype::Bfloat16),
        14 => Ok(Dtype::Complex64),
        _ => Err(std::io::Error::other("unknown array dtype")),
    }
}
const fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
    }
}
fn le_bytes<T: Copy, F: Fn(T) -> Vec<u8>>(slice: &[T], encode: F) -> Vec<u8> {
    slice.iter().copied().flat_map(encode).collect()
}
fn array_bytes(array: &Array) -> std::io::Result<Vec<u8>> {
    match array.dtype() {
        Dtype::Bool => Ok(array
            .as_slice::<bool>()
            .iter()
            .map(|&v| u8::from(v))
            .collect()),
        Dtype::Uint8 => Ok(array.as_slice::<u8>().to_vec()),
        Dtype::Int8 => Ok(array.as_slice::<i8>().iter().map(|&v| v as u8).collect()),
        Dtype::Uint16 => Ok(le_bytes(array.as_slice::<u16>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Int16 => Ok(le_bytes(array.as_slice::<i16>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Uint32 => Ok(le_bytes(array.as_slice::<u32>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Int32 => Ok(le_bytes(array.as_slice::<i32>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Uint64 => Ok(le_bytes(array.as_slice::<u64>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Int64 => Ok(le_bytes(array.as_slice::<i64>(), |v| {
            v.to_le_bytes().to_vec()
        })),
        Dtype::Float16 => Ok(le_bytes(array.as_slice::<f16>(), |v| {
            v.to_bits().to_le_bytes().to_vec()
        })),
        Dtype::Bfloat16 => Ok(le_bytes(array.as_slice::<bf16>(), |v| {
            v.to_bits().to_le_bytes().to_vec()
        })),
        Dtype::Float32 => Ok(le_bytes(array.as_slice::<f32>(), |v| {
            v.to_bits().to_le_bytes().to_vec()
        })),
        Dtype::Float64 => Ok(le_bytes(array.as_slice::<f64>(), |v| {
            v.to_bits().to_le_bytes().to_vec()
        })),
        Dtype::Complex64 => Ok(array
            .as_slice::<complex64>()
            .iter()
            .flat_map(|v| {
                [v.re.to_bits().to_le_bytes(), v.im.to_bits().to_le_bytes()]
                    .into_iter()
                    .flatten()
            })
            .collect()),
    }
}
fn words<const N: usize>(bytes: &[u8]) -> Vec<[u8; N]> {
    bytes
        .chunks_exact(N)
        .map(|b| b.try_into().unwrap_or([0; N]))
        .collect()
}
fn array_from_bytes(dtype: Dtype, shape: &[i32], bytes: &[u8]) -> std::io::Result<Array> {
    Ok(match dtype {
        Dtype::Bool => Array::from_slice(&bytes.iter().map(|&v| v != 0).collect::<Vec<_>>(), shape),
        Dtype::Uint8 => Array::from_slice(bytes, shape),
        Dtype::Int8 => {
            Array::from_slice(&bytes.iter().map(|&v| v as i8).collect::<Vec<_>>(), shape)
        }
        Dtype::Uint16 => Array::from_slice(
            &words::<2>(bytes)
                .into_iter()
                .map(u16::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Int16 => Array::from_slice(
            &words::<2>(bytes)
                .into_iter()
                .map(i16::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Uint32 => Array::from_slice(
            &words::<4>(bytes)
                .into_iter()
                .map(u32::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Int32 => Array::from_slice(
            &words::<4>(bytes)
                .into_iter()
                .map(i32::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Uint64 => Array::from_slice(
            &words::<8>(bytes)
                .into_iter()
                .map(u64::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Int64 => Array::from_slice(
            &words::<8>(bytes)
                .into_iter()
                .map(i64::from_le_bytes)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Float16 => Array::from_slice(
            &words::<2>(bytes)
                .into_iter()
                .map(|v| f16::from_bits(u16::from_le_bytes(v)))
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Bfloat16 => Array::from_slice(
            &words::<2>(bytes)
                .into_iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(v)))
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Float32 => Array::from_slice(
            &words::<4>(bytes)
                .into_iter()
                .map(|v| f32::from_bits(u32::from_le_bytes(v)))
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Float64 => Array::from_slice_f64(
            &words::<8>(bytes)
                .into_iter()
                .map(|v| f64::from_bits(u64::from_le_bytes(v)))
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Complex64 => Array::from_slice(
            &bytes
                .chunks_exact(8)
                .map(|v| complex64 {
                    re: f32::from_bits(u32::from_le_bytes(v[..4].try_into().unwrap_or([0; 4]))),
                    im: f32::from_bits(u32::from_le_bytes(v[4..].try_into().unwrap_or([0; 4]))),
                })
                .collect::<Vec<_>>(),
            shape,
        ),
    })
}
fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn key(tokens: &[u32], block_size: usize) -> String {
    let n = tokens.len() / block_size * block_size;
    let bytes: Vec<u8> = tokens[..n].iter().flat_map(|t| t.to_le_bytes()).collect();
    hex_lower(&sha256(&bytes))
}
fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}
/// Build a temp-file path that is unique per writer, so two processes (or two
/// writers in the same process) sharing a store directory cannot clobber each
/// other's in-progress write of the same key.
fn unique_tmp_path(path: &Path) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("entry");
    path.with_file_name(format!("{file_name}.tmp.{pid}.{nanos}.{counter}"))
}
/// Whether `path` is a temp file left by an in-progress or interrupted write,
/// covering both the unique per-writer naming (`<name>.tmp.<pid>.<n>.<c>`)
/// and the plain `*.tmp` suffix.
fn is_tmp_file(path: &Path) -> bool {
    path.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
        n.rsplit_once('.')
            .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("tmp"))
            || n.contains(".tmp.")
    })
}
fn write_u32<W: Write>(w: &mut W, n: u32) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn write_u64<W: Write>(w: &mut W, n: u64) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}
fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    write_u32(
        w,
        u32::try_from(s.len()).map_err(|_| std::io::Error::other("string too long"))?,
    )?;
    w.write_all(s.as_bytes())
}
fn read_string<R: Read>(r: &mut R) -> Option<String> {
    let n = usize::try_from(read_u32(r)?).ok()?;
    if n > 4096 {
        return None;
    }
    let mut b = vec![0; n];
    r.read_exact(&mut b).ok()?;
    String::from_utf8(b).ok()
}
fn touch(path: &Path) -> std::io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_times(fs::FileTimes::new().set_modified(std::time::SystemTime::now()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_prefix_cache::{DEFAULT_BLOCK_SIZE, PagedPrefixCache};
    use higgs_models::{
        AnyCache,
        cache::{KeyValueCache, SteppingKeyValueCache},
    };
    use mlx_rs::Array;
    use tempfile::tempdir;
    fn identity() -> StoreIdentity {
        StoreIdentity::for_tests()
    }
    #[test]
    fn sha256_digest_bytes_and_lowercase_hex_are_stable() {
        assert_eq!(
            hex_lower(&sha256(b"higgs")),
            "a93f7a91f88ee389a5147b54b22968c2521a59e41b1bf98ff0d22d7b5a704a42"
        );
    }
    #[test]
    fn round_trip_preserves_dense_payload() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let payload = DensePayload::test_payload();
        store.store_payload(&tokens, 2, &payload).unwrap();
        assert_eq!(store.load_payload(&tokens).unwrap(), payload);
    }
    #[test]
    fn round_trip_preserves_dense_kv_arrays() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let keys = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = Array::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        let cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
        )]);

        store.store_cache(&tokens, 2, &cache).unwrap();
        let (_, restored) = store.load_cache(&tokens, 2, &cache).unwrap().unwrap();
        let AnyCache::KV(layers) = restored else {
            panic!("expected KV cache")
        };
        let layer = layers[0].as_ref().unwrap();
        assert_eq!(
            layer.keys().unwrap().as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            layer.values().unwrap().as_slice::<f32>(),
            &[5.0, 6.0, 7.0, 8.0]
        );
    }
    #[test]
    fn round_trip_preserves_f16_dtype_and_bits() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let values = [
            f16::from_bits(0x3c00),
            f16::from_bits(0xc000),
            f16::from_bits(0x7bff),
            f16::from_bits(0x0001),
        ];
        let cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(
                Array::from_slice(&values, &[1, 1, 4, 1]),
                Array::from_slice(&values, &[1, 1, 4, 1]),
            )
            .unwrap(),
        )]);
        store.store_cache(&tokens, 2, &cache).unwrap();
        let (_, restored) = store.load_cache(&tokens, 2, &cache).unwrap().unwrap();
        let AnyCache::KV(layers) = restored else {
            panic!("expected KV cache")
        };
        let keys = layers[0].as_ref().unwrap().keys().unwrap();
        assert_eq!(keys.dtype(), Dtype::Float16);
        assert_eq!(
            keys.as_slice::<f16>()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn restart_restores_an_unaligned_prompt_into_the_paged_prefix_cache() {
        let dir = tempdir().unwrap();
        let prompt: Vec<u32> = (0..99).collect();
        let cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(
                Array::zeros::<f32>(&[1, 2, 99, 8]).unwrap(),
                Array::zeros::<f32>(&[1, 2, 99, 8]).unwrap(),
            )
            .unwrap(),
        )]);

        DiskPrefixStore::new(dir.path(), 1024 * 1024, identity())
            .unwrap()
            .store_cache(&prompt, u32::try_from(DEFAULT_BLOCK_SIZE).unwrap(), &cache)
            .unwrap();

        let restarted_store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let prototype = AnyCache::KV(vec![Some(SteppingKeyValueCache::new())]);
        let Some((prefix_len, restored)) = restarted_store
            .load_cache(
                &prompt,
                u32::try_from(DEFAULT_BLOCK_SIZE).unwrap(),
                &prototype,
            )
            .unwrap()
        else {
            panic!("restart should restore the block-aligned prefix");
        };
        let mut paged_cache = PagedPrefixCache::new(1, DEFAULT_BLOCK_SIZE);
        paged_cache.store(&prompt[..prefix_len], &restored);
        let Some(consumed) = paged_cache.find_longest_prefix(&prompt) else {
            panic!("materialized disk cache should be consumable");
        };

        assert_eq!(consumed.prefix_len, 96);
        let AnyCache::KV(layers) = consumed.cache else {
            panic!("expected KV cache")
        };
        assert_eq!(layers[0].as_ref().unwrap().offset(), 96);
    }
    #[test]
    fn invalid_headers_are_clean_misses() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        for corruption in [
            Corruption::Truncate,
            Corruption::Model,
            Corruption::Quant,
            Corruption::Tokenizer,
            Corruption::Magic,
            Corruption::Weights,
        ] {
            store
                .store_payload(&tokens, 2, &DensePayload::test_payload())
                .unwrap();
            store.corrupt_for_test(&tokens, corruption).unwrap();
            assert!(store.load_payload(&tokens).is_none());
        }
    }
    #[test]
    fn flipped_byte_in_tensor_payload_is_a_clean_miss() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let keys = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = Array::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        let cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
        )]);
        store.store_cache(&tokens, 2, &cache).unwrap();

        let path = store
            .files()
            .unwrap()
            .into_iter()
            .next()
            .map(|(p, _)| p)
            .unwrap();
        let mut bytes = fs::read(&path).unwrap();
        // Flip the last byte: it necessarily falls inside the serialized
        // tensor payload (the arrays occupy the tail of the file).
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(&path, bytes).unwrap();

        assert!(store.load_cache(&tokens, 2, &cache).unwrap().is_none());
    }
    #[test]
    fn absurd_payload_len_is_a_clean_miss_not_a_panic() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let tokens = vec![1, 2, 3, 4];
        let keys = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = Array::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        let cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
        )]);
        store.store_cache(&tokens, 2, &cache).unwrap();

        let path = store
            .files()
            .unwrap()
            .into_iter()
            .next()
            .map(|(p, _)| p)
            .unwrap();
        let mut bytes = fs::read(&path).unwrap();
        // payload_len (u64 LE) is the last 8 bytes of the header, i.e. the 8
        // bytes immediately preceding the tensor payload. `identity()` uses
        // the fixed test model_id/quant strings, so the header layout (and
        // therefore this offset) is deterministic:
        // magic(4) + version(4) + model_id(4+10) + quant(4+5) +
        // config_hash(32) + tokenizer_hash(32) + chat_template_hash(32) +
        // weights_hash(32) + block_size(4) + token_count(4) +
        // payload_checksum(32) = 199, then payload_len occupies 199..207.
        let payload_len_offset = 199;
        bytes[payload_len_offset..payload_len_offset + 8]
            .copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        fs::write(&path, bytes).unwrap();

        // Must not panic (e.g. from an oversized allocation) and must be a
        // clean cache miss.
        assert!(store.load_cache(&tokens, 2, &cache).unwrap().is_none());
    }
    #[test]
    fn longest_stored_prefix_is_selected_deterministically() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        let block_size = u32::try_from(DEFAULT_BLOCK_SIZE).unwrap();
        let conversation: Vec<u32> = (0..u32::try_from(4 * DEFAULT_BLOCK_SIZE).unwrap()).collect();

        let short = &conversation[..2 * DEFAULT_BLOCK_SIZE];
        let long = &conversation[..3 * DEFAULT_BLOCK_SIZE];
        let short_len = i32::try_from(short.len()).unwrap();
        let long_len = i32::try_from(long.len()).unwrap();
        let short_cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(
                Array::zeros::<f32>(&[1, 1, short_len, 1]).unwrap(),
                Array::zeros::<f32>(&[1, 1, short_len, 1]).unwrap(),
            )
            .unwrap(),
        )]);
        let long_cache = AnyCache::KV(vec![Some(
            SteppingKeyValueCache::from_arrays(
                Array::zeros::<f32>(&[1, 1, long_len, 1]).unwrap(),
                Array::zeros::<f32>(&[1, 1, long_len, 1]).unwrap(),
            )
            .unwrap(),
        )]);
        store.store_cache(short, block_size, &short_cache).unwrap();
        store.store_cache(long, block_size, &long_cache).unwrap();

        // A request extending the long prefix should restore the long
        // prefix, not the shorter one, even though both are compatible.
        let prototype = AnyCache::KV(vec![Some(SteppingKeyValueCache::new())]);
        let (prefix_len, _) = store
            .load_cache(&conversation, block_size, &prototype)
            .unwrap()
            .unwrap();
        assert_eq!(prefix_len, 3 * DEFAULT_BLOCK_SIZE);
    }
    #[test]
    fn eviction_removes_least_recently_touched_file() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 700, identity()).unwrap();
        let first = vec![1, 2];
        let second = vec![3, 4];
        store
            .store_payload(&first, 2, &DensePayload::sized(100))
            .unwrap();
        store
            .store_payload(&second, 2, &DensePayload::sized(100))
            .unwrap();
        assert!(store.load_payload(&first).is_some());
        store
            .store_payload(&[5, 6], 2, &DensePayload::sized(100))
            .unwrap();
        assert!(store.load_payload(&first).is_some());
        assert!(store.load_payload(&second).is_none());
    }
    #[test]
    fn leftover_temporary_file_is_ignored_and_cleaned() {
        let dir = tempdir().unwrap();
        let store = DiskPrefixStore::new(dir.path(), 1024 * 1024, identity()).unwrap();
        fs::write(dir.path().join("orphan.tmp"), b"partial").unwrap();
        store.cleanup_temps().unwrap();
        assert!(!dir.path().join("orphan.tmp").exists());
    }
}
