// SPDX-License-Identifier: Apache-2.0
//! Append-only disk storage for persisted prefix KV cache blocks.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use half::f16;
use memmap2::{Mmap, MmapOptions};

const FILE_MAGIC: [u8; 8] = *b"HIGGSKV\0";
const ENTRY_MAGIC: [u8; 4] = *b"BLK1";
const VERSION: u32 = 1;
const FILE_HEADER_LEN: usize = 32;
const ENTRY_HEADER_LEN: usize = 44;

#[derive(Debug, thiserror::Error)]
pub enum DiskCacheError {
    #[error("disk cache io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("disk cache format error: {0}")]
    Format(String),
    #[error("disk cache does not support this cache: {0}")]
    Unsupported(&'static str),
    #[error("disk cache value is too large: {0}")]
    Overflow(&'static str),
    #[error("disk cache MLX error: {0}")]
    Mlx(String),
}

/// Dense K/V data for one persisted block in f16 format.
#[derive(Debug, Clone)]
pub struct DiskCacheBlock {
    pub k: Vec<f16>,
    pub v: Vec<f16>,
}

/// One model layer in a persisted prefix snapshot.
#[derive(Debug, Clone)]
pub struct DiskCacheLayer {
    pub blocks: Vec<DiskCacheBlock>,
}

/// Full persisted prefix snapshot.
#[derive(Debug, Clone)]
pub struct DiskCacheSnapshot {
    pub token_count: usize,
    pub token_hash: u64,
    pub layers: Vec<DiskCacheLayer>,
}

#[derive(Debug, Clone)]
pub struct DiskCacheFileHeader {
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DiskCacheEntryMetadata {
    pub token_count: usize,
    pub token_hash: u64,
}

#[derive(Debug, Clone, Copy)]
struct EntryIndex {
    offset: usize,
    metadata: DiskCacheEntryMetadata,
}

#[derive(Debug, Clone, Copy)]
struct EntryHeader {
    session_id: u64,
    token_count: usize,
    token_hash: u64,
    layer_count: usize,
    block_count: usize,
    payload_len: usize,
}

/// Append-only binary storage for dense KV prefix snapshots.
#[derive(Debug)]
pub struct DiskStorage {
    file: File,
    header: DiskCacheFileHeader,
    index: HashMap<u64, EntryIndex>,
}

impl DiskStorage {
    pub fn open<P: AsRef<Path>>(
        path: P,
        block_size: usize,
        num_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, DiskCacheError> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path_ref)?;

        let expected_header = DiskCacheFileHeader {
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
        };

        if file.metadata()?.len() == 0 {
            write_file_header(&mut file, &expected_header)?;
            file.flush()?;
        }

        let map = map_file(&file)?;
        let found_header = read_file_header(&map)?;
        validate_header(&expected_header, &found_header)?;
        let index = scan_index(&map);

        Ok(Self {
            file,
            header: found_header,
            index,
        })
    }

    pub const fn header(&self) -> &DiskCacheFileHeader {
        &self.header
    }

    pub fn snapshot_metadata(&self, session_id: u64) -> Option<DiskCacheEntryMetadata> {
        self.index.get(&session_id).map(|entry| entry.metadata)
    }

    pub fn save_blocks(
        &mut self,
        session_id: u64,
        token_hash: u64,
        token_count: usize,
        layers: &[DiskCacheLayer],
    ) -> Result<(), DiskCacheError> {
        if token_count == 0 {
            return Err(DiskCacheError::Format(
                "token_count must be greater than zero".to_owned(),
            ));
        }
        let block_count = token_count / self.header.block_size;
        if block_count == 0 || block_count.saturating_mul(self.header.block_size) != token_count {
            return Err(DiskCacheError::Format(
                "token_count must be block-aligned".to_owned(),
            ));
        }
        if block_count > self.header.num_blocks {
            return Err(DiskCacheError::Format(format!(
                "snapshot has {block_count} blocks, limit is {}",
                self.header.num_blocks
            )));
        }

        let block_elems = self.block_elems()?;
        let mut payload = Vec::new();
        for layer in layers {
            if layer.blocks.is_empty() {
                payload.push(0);
                continue;
            }
            if layer.blocks.len() != block_count {
                return Err(DiskCacheError::Format(
                    "layer block count does not match token_count".to_owned(),
                ));
            }
            payload.push(1);
            for block in &layer.blocks {
                if block.k.len() != block_elems || block.v.len() != block_elems {
                    return Err(DiskCacheError::Format(
                        "block payload length does not match cache layout".to_owned(),
                    ));
                }
                push_f16_slice(&mut payload, &block.k);
                push_f16_slice(&mut payload, &block.v);
            }
        }

        let append_offset = usize::try_from(self.file.metadata()?.len())
            .map_err(|_| DiskCacheError::Overflow("file length"))?;
        let mut header_bytes = Vec::with_capacity(ENTRY_HEADER_LEN);
        header_bytes.extend_from_slice(&ENTRY_MAGIC);
        header_bytes.extend_from_slice(&session_id.to_le_bytes());
        header_bytes.extend_from_slice(&u64_from_usize(token_count, "token_count")?.to_le_bytes());
        header_bytes.extend_from_slice(&token_hash.to_le_bytes());
        header_bytes.extend_from_slice(&u32_from_usize(layers.len(), "layer_count")?.to_le_bytes());
        header_bytes.extend_from_slice(&u32_from_usize(block_count, "block_count")?.to_le_bytes());
        header_bytes
            .extend_from_slice(&u64_from_usize(payload.len(), "payload_len")?.to_le_bytes());

        self.file.write_all(&header_bytes)?;
        self.file.write_all(&payload)?;
        self.file.flush()?;
        self.index.insert(
            session_id,
            EntryIndex {
                offset: append_offset,
                metadata: DiskCacheEntryMetadata {
                    token_count,
                    token_hash,
                },
            },
        );
        Ok(())
    }

    pub fn load_blocks(
        &self,
        session_id: u64,
    ) -> Result<Option<DiskCacheSnapshot>, DiskCacheError> {
        let Some(entry_index) = self.index.get(&session_id).copied() else {
            return Ok(None);
        };
        let map = map_file(&self.file)?;
        let (entry_header, payload_start) = read_entry_header(&map, entry_index.offset)?;
        if entry_header.session_id != session_id {
            return Ok(None);
        }
        let payload_end = payload_start
            .checked_add(entry_header.payload_len)
            .ok_or(DiskCacheError::Overflow("payload end"))?;
        let payload = map
            .get(payload_start..payload_end)
            .ok_or_else(|| DiskCacheError::Format("entry payload is truncated".to_owned()))?;
        let layers = read_layers(payload, &self.header, &entry_header)?;
        Ok(Some(DiskCacheSnapshot {
            token_count: entry_header.token_count,
            token_hash: entry_header.token_hash,
            layers,
        }))
    }

    fn block_elems(&self) -> Result<usize, DiskCacheError> {
        self.header
            .block_size
            .checked_mul(self.header.num_kv_heads)
            .and_then(|value| value.checked_mul(self.header.head_dim))
            .ok_or(DiskCacheError::Overflow("block elements"))
    }
}

fn write_file_header(file: &mut File, header: &DiskCacheFileHeader) -> Result<(), DiskCacheError> {
    file.write_all(&FILE_MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&u32_from_usize(header.block_size, "block_size")?.to_le_bytes())?;
    file.write_all(&u64_from_usize(header.num_blocks, "num_blocks")?.to_le_bytes())?;
    file.write_all(&u32_from_usize(header.num_kv_heads, "num_kv_heads")?.to_le_bytes())?;
    file.write_all(&u32_from_usize(header.head_dim, "head_dim")?.to_le_bytes())?;
    Ok(())
}

fn read_file_header(bytes: &[u8]) -> Result<DiskCacheFileHeader, DiskCacheError> {
    let mut cursor = 0;
    let magic = take_array::<8>(bytes, &mut cursor)?;
    if magic != FILE_MAGIC {
        return Err(DiskCacheError::Format(
            "invalid disk cache magic".to_owned(),
        ));
    }
    let version = u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?);
    if version != VERSION {
        return Err(DiskCacheError::Format(format!(
            "unsupported disk cache version {version}"
        )));
    }
    let block_size = usize_from_u32(u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?))?;
    let num_blocks = usize_from_u64(u64::from_le_bytes(take_array::<8>(bytes, &mut cursor)?))?;
    let num_kv_heads = usize_from_u32(u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?))?;
    let head_dim = usize_from_u32(u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?))?;
    Ok(DiskCacheFileHeader {
        block_size,
        num_blocks,
        num_kv_heads,
        head_dim,
    })
}

fn validate_header(
    expected: &DiskCacheFileHeader,
    found: &DiskCacheFileHeader,
) -> Result<(), DiskCacheError> {
    if expected.block_size != found.block_size
        || expected.num_kv_heads != found.num_kv_heads
        || expected.head_dim != found.head_dim
    {
        return Err(DiskCacheError::Format(format!(
            "disk cache layout mismatch: expected block_size={} heads={} head_dim={}, found block_size={} heads={} head_dim={}",
            expected.block_size,
            expected.num_kv_heads,
            expected.head_dim,
            found.block_size,
            found.num_kv_heads,
            found.head_dim
        )));
    }
    Ok(())
}

fn scan_index(bytes: &[u8]) -> HashMap<u64, EntryIndex> {
    let mut index = HashMap::new();
    let mut cursor = FILE_HEADER_LEN;
    while cursor < bytes.len() {
        let entry_offset = cursor;
        let Ok((entry_header, payload_start)) = read_entry_header(bytes, cursor) else {
            break;
        };
        let Some(payload_end) = payload_start.checked_add(entry_header.payload_len) else {
            break;
        };
        if payload_end > bytes.len() {
            break;
        }
        index.insert(
            entry_header.session_id,
            EntryIndex {
                offset: entry_offset,
                metadata: DiskCacheEntryMetadata {
                    token_count: entry_header.token_count,
                    token_hash: entry_header.token_hash,
                },
            },
        );
        cursor = payload_end;
    }
    index
}

fn read_entry_header(bytes: &[u8], offset: usize) -> Result<(EntryHeader, usize), DiskCacheError> {
    let mut cursor = offset;
    let magic = take_array::<4>(bytes, &mut cursor)?;
    if magic != ENTRY_MAGIC {
        return Err(DiskCacheError::Format("invalid entry magic".to_owned()));
    }
    let session_id = u64::from_le_bytes(take_array::<8>(bytes, &mut cursor)?);
    let token_count = usize_from_u64(u64::from_le_bytes(take_array::<8>(bytes, &mut cursor)?))?;
    let token_hash = u64::from_le_bytes(take_array::<8>(bytes, &mut cursor)?);
    let layer_count = usize_from_u32(u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?))?;
    let block_count = usize_from_u32(u32::from_le_bytes(take_array::<4>(bytes, &mut cursor)?))?;
    let payload_len = usize_from_u64(u64::from_le_bytes(take_array::<8>(bytes, &mut cursor)?))?;
    Ok((
        EntryHeader {
            session_id,
            token_count,
            token_hash,
            layer_count,
            block_count,
            payload_len,
        },
        cursor,
    ))
}

fn read_layers(
    payload: &[u8],
    file_header: &DiskCacheFileHeader,
    entry_header: &EntryHeader,
) -> Result<Vec<DiskCacheLayer>, DiskCacheError> {
    let block_elems = file_header
        .block_size
        .checked_mul(file_header.num_kv_heads)
        .and_then(|value| value.checked_mul(file_header.head_dim))
        .ok_or(DiskCacheError::Overflow("block elements"))?;
    let mut cursor = 0;
    let mut layers = Vec::with_capacity(entry_header.layer_count);
    for _ in 0..entry_header.layer_count {
        let kind = u8::from_le_bytes(take_array::<1>(payload, &mut cursor)?);
        if kind == 0 {
            layers.push(DiskCacheLayer { blocks: Vec::new() });
            continue;
        }
        if kind != 1 {
            return Err(DiskCacheError::Format(format!("unknown layer kind {kind}")));
        }
        let mut blocks = Vec::with_capacity(entry_header.block_count);
        for _ in 0..entry_header.block_count {
            let k = take_f16_vec(payload, &mut cursor, block_elems)?;
            let v = take_f16_vec(payload, &mut cursor, block_elems)?;
            blocks.push(DiskCacheBlock { k, v });
        }
        layers.push(DiskCacheLayer { blocks });
    }
    if cursor != payload.len() {
        return Err(DiskCacheError::Format(
            "entry payload has trailing bytes".to_owned(),
        ));
    }
    Ok(layers)
}

fn take_f16_vec(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<f16>, DiskCacheError> {
    let byte_len = len
        .checked_mul(2)
        .ok_or(DiskCacheError::Overflow("f16 byte length"))?;
    let slice = take_slice(bytes, cursor, byte_len)?;
    slice
        .chunks_exact(2)
        .map(|chunk| {
            let raw: [u8; 2] = chunk
                .try_into()
                .map_err(|_| DiskCacheError::Format("invalid f16 chunk length".to_owned()))?;
            Ok(f16::from_bits(u16::from_le_bytes(raw)))
        })
        .collect()
}

fn push_f16_slice(out: &mut Vec<u8>, values: &[f16]) {
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn take_array<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], DiskCacheError> {
    let slice = take_slice(bytes, cursor, N)?;
    slice
        .try_into()
        .map_err(|_| DiskCacheError::Format("invalid fixed-width field".to_owned()))
}

fn take_slice<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], DiskCacheError> {
    let end = cursor
        .checked_add(len)
        .ok_or(DiskCacheError::Overflow("cursor"))?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| DiskCacheError::Format("unexpected end of file".to_owned()))?;
    *cursor = end;
    Ok(slice)
}

#[allow(unsafe_code)]
fn map_file(file: &File) -> Result<Mmap, DiskCacheError> {
    unsafe { MmapOptions::new().map(file).map_err(DiskCacheError::Io) }
}

fn u32_from_usize(value: usize, label: &'static str) -> Result<u32, DiskCacheError> {
    u32::try_from(value).map_err(|_| DiskCacheError::Overflow(label))
}

fn u64_from_usize(value: usize, label: &'static str) -> Result<u64, DiskCacheError> {
    u64::try_from(value).map_err(|_| DiskCacheError::Overflow(label))
}

fn usize_from_u32(value: u32) -> Result<usize, DiskCacheError> {
    usize::try_from(value).map_err(|_| DiskCacheError::Overflow("u32 to usize"))
}

fn usize_from_u64(value: u64) -> Result<usize, DiskCacheError> {
    usize::try_from(value).map_err(|_| DiskCacheError::Overflow("u64 to usize"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_blocks_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prefix.bin");
        let mut storage = DiskStorage::open(&path, 2, 16, 2, 2).unwrap();

        let block = DiskCacheBlock {
            k: vec![f16::from_f32(1.0); 8],
            v: vec![f16::from_f32(2.0); 8],
        };
        let layers = vec![DiskCacheLayer {
            blocks: vec![block.clone(), block],
        }];

        storage.save_blocks(7, 99, 4, &layers).unwrap();
        drop(storage);

        let reopened_storage = DiskStorage::open(&path, 2, 16, 2, 2).unwrap();
        let snapshot = reopened_storage.load_blocks(7).unwrap().unwrap();
        assert_eq!(snapshot.token_count, 4);
        assert_eq!(snapshot.token_hash, 99);
        assert_eq!(snapshot.layers.len(), 1);
        assert_eq!(snapshot.layers[0].blocks.len(), 2);
        assert_eq!(snapshot.layers[0].blocks[0].k[0], f16::from_f32(1.0));
        assert_eq!(snapshot.layers[0].blocks[0].v[0], f16::from_f32(2.0));
    }

    #[test]
    fn latest_entry_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prefix.bin");
        let mut storage = DiskStorage::open(&path, 2, 16, 1, 1).unwrap();

        let first = vec![DiskCacheLayer {
            blocks: vec![DiskCacheBlock {
                k: vec![f16::from_f32(1.0); 2],
                v: vec![f16::from_f32(1.0); 2],
            }],
        }];
        let second = vec![DiskCacheLayer {
            blocks: vec![DiskCacheBlock {
                k: vec![f16::from_f32(3.0); 2],
                v: vec![f16::from_f32(4.0); 2],
            }],
        }];

        storage.save_blocks(7, 1, 2, &first).unwrap();
        storage.save_blocks(7, 2, 2, &second).unwrap();

        let snapshot = storage.load_blocks(7).unwrap().unwrap();
        assert_eq!(snapshot.token_hash, 2);
        assert_eq!(snapshot.layers[0].blocks[0].k[0], f16::from_f32(3.0));
        assert_eq!(snapshot.layers[0].blocks[0].v[0], f16::from_f32(4.0));
    }
}
