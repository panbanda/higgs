//! Durable, model-bound prefix-cache entries.
//!
//! Files are atomically written below `<dir>/<first-two-key-hex>/<key>.hkv`.
//! File mtime is the v1 frecency signal: hits touch files and stores evict the
//! least recently touched entries until the configured budget is respected.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"HKV1";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreIdentity {
    pub model_id: String,
    pub quant: String,
    pub config_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub chat_template_hash: [u8; 32],
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
        }
    }
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
        let tmp = path.with_extension("tmp");
        let mut file = File::create(&tmp)?;
        file.write_all(MAGIC)?;
        write_u32(&mut file, VERSION)?;
        write_string(&mut file, &self.identity.model_id)?;
        write_string(&mut file, &self.identity.quant)?;
        file.write_all(&self.identity.config_hash)?;
        file.write_all(&self.identity.tokenizer_hash)?;
        file.write_all(&self.identity.chat_template_hash)?;
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
                    if child.path().extension().and_then(|e| e.to_str()) == Some("tmp") {
                        fs::remove_file(child.path())?;
                    }
                }
            } else if entry.path().extension().and_then(|e| e.to_str()) == Some("tmp") {
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
}
fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn key(tokens: &[u32], block_size: usize) -> String {
    let n = tokens.len() / block_size * block_size;
    let bytes: Vec<u8> = tokens[..n].iter().flat_map(|t| t.to_le_bytes()).collect();
    hex(&sha256(&bytes))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn write_u32(w: &mut File, n: u32) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn write_u64(w: &mut File, n: u64) -> std::io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn read_u32(r: &mut File) -> Option<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}
fn read_u64(r: &mut File) -> Option<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}
fn write_string(w: &mut File, s: &str) -> std::io::Result<()> {
    write_u32(
        w,
        u32::try_from(s.len()).map_err(|_| std::io::Error::other("string too long"))?,
    )?;
    w.write_all(s.as_bytes())
}
fn read_string(r: &mut File) -> Option<String> {
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
    use tempfile::tempdir;
    fn identity() -> StoreIdentity {
        StoreIdentity::for_tests()
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
        ] {
            store
                .store_payload(&tokens, 2, &DensePayload::test_payload())
                .unwrap();
            store.corrupt_for_test(&tokens, corruption).unwrap();
            assert!(store.load_payload(&tokens).is_none());
        }
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
