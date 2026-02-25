use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::config::MetricsLogConfig;

pub struct MetricsLogger {
    path: PathBuf,
    max_size: u64,
    max_files: u32,
    writer: BufWriter<File>,
}

impl MetricsLogger {
    pub fn new(config: &MetricsLogConfig) -> io::Result<Self> {
        let path = PathBuf::from(&config.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            max_size: config.max_size_mb * 1024 * 1024,
            max_files: config.max_files,
            writer: BufWriter::new(file),
        })
    }

    pub fn write_line(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.writer, "{line}")?;
        self.writer.flush()?;
        self.maybe_rotate()
    }

    fn maybe_rotate(&mut self) -> io::Result<()> {
        if self.max_files == 0 {
            return Ok(());
        }
        let size = fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        if size < self.max_size {
            return Ok(());
        }
        self.rotate()
    }

    fn rotate(&mut self) -> io::Result<()> {
        let oldest = rotated_path(&self.path, self.max_files);
        if oldest.exists() {
            fs::remove_file(&oldest)?;
        }
        for i in (1..self.max_files).rev() {
            let from = rotated_path(&self.path, i);
            let to = rotated_path(&self.path, i + 1);
            if from.exists() {
                fs::rename(&from, &to)?;
            }
        }
        let first_rotated = rotated_path(&self.path, 1);
        fs::rename(&self.path, &first_rotated)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        Ok(())
    }
}

pub(crate) fn rotated_path(base: &Path, index: u32) -> PathBuf {
    let name = base.file_name().unwrap_or_default().to_string_lossy();
    base.with_file_name(format!("{name}.{index}"))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn test_config(dir: &Path, max_size_mb: u64, max_files: u32) -> MetricsLogConfig {
        MetricsLogConfig {
            enabled: true,
            path: dir.join("metrics.jsonl").to_string_lossy().to_string(),
            max_size_mb,
            max_files,
        }
    }

    #[test]
    fn test_creates_parent_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a/b/c");
        let config = MetricsLogConfig {
            enabled: true,
            path: nested.join("metrics.jsonl").to_string_lossy().to_string(),
            max_size_mb: 1,
            max_files: 3,
        };
        let _logger = MetricsLogger::new(&config).unwrap();
        assert!(nested.exists());
    }

    #[test]
    fn test_write_line_creates_file_content() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path(), 1, 3);
        let log_path = PathBuf::from(&config.path);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line(r#"{"event":"test"}"#).unwrap();

        let content = fs::read_to_string(&log_path).unwrap();
        assert_eq!(content.trim(), r#"{"event":"test"}"#);
    }

    #[test]
    fn test_multiple_writes_append() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path(), 1, 3);
        let log_path = PathBuf::from(&config.path);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line("line1").unwrap();
        logger.write_line("line2").unwrap();
        logger.write_line("line3").unwrap();

        let content = fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
        assert_eq!(lines[2], "line3");
    }

    #[test]
    fn test_no_rotation_under_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path(), 1, 3);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line("small").unwrap();

        let rotated = rotated_path(&PathBuf::from(&config.path), 1);
        assert!(!rotated.exists());
    }

    #[test]
    fn test_rotation_when_size_exceeded() {
        let tmp = tempfile::tempdir().unwrap();
        // Use a very small max_size_mb -- 0 means 0 bytes, so any write triggers rotation.
        // Instead, write enough to exceed a small limit. We'll set max_size_mb to 0 which
        // means max_size = 0, so the first write will exceed it.
        let config = MetricsLogConfig {
            enabled: true,
            path: tmp
                .path()
                .join("metrics.jsonl")
                .to_string_lossy()
                .to_string(),
            max_size_mb: 0,
            max_files: 3,
        };
        let log_path = PathBuf::from(&config.path);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line("first").unwrap();

        // After writing "first\n", size > 0 = max_size, so rotation should have happened.
        let rotated_1 = rotated_path(&log_path, 1);
        assert!(rotated_1.exists(), "rotated file .1 should exist");

        let rotated_content = fs::read_to_string(&rotated_1).unwrap();
        assert_eq!(rotated_content.trim(), "first");

        // The main file should now be empty (new file created after rotation).
        let main_content = fs::read_to_string(&log_path).unwrap();
        assert!(main_content.is_empty());
    }

    #[test]
    fn test_rotation_shifts_files() {
        let tmp = tempfile::tempdir().unwrap();
        let config = MetricsLogConfig {
            enabled: true,
            path: tmp
                .path()
                .join("metrics.jsonl")
                .to_string_lossy()
                .to_string(),
            max_size_mb: 0,
            max_files: 3,
        };
        let log_path = PathBuf::from(&config.path);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line("first").unwrap();
        logger.write_line("second").unwrap();
        logger.write_line("third").unwrap();

        // After 3 rotations: .1 = "third", .2 = "second", .3 = "first"
        let content_1 = fs::read_to_string(rotated_path(&log_path, 1)).unwrap();
        let content_2 = fs::read_to_string(rotated_path(&log_path, 2)).unwrap();
        let content_3 = fs::read_to_string(rotated_path(&log_path, 3)).unwrap();

        assert_eq!(content_1.trim(), "third");
        assert_eq!(content_2.trim(), "second");
        assert_eq!(content_3.trim(), "first");
    }

    #[test]
    fn test_oldest_file_deleted_on_rotation() {
        let tmp = tempfile::tempdir().unwrap();
        let config = MetricsLogConfig {
            enabled: true,
            path: tmp
                .path()
                .join("metrics.jsonl")
                .to_string_lossy()
                .to_string(),
            max_size_mb: 0,
            max_files: 2,
        };
        let log_path = PathBuf::from(&config.path);

        let mut logger = MetricsLogger::new(&config).unwrap();
        logger.write_line("first").unwrap();
        logger.write_line("second").unwrap();
        logger.write_line("third").unwrap();

        // max_files=2, so .1 and .2 exist, but "first" was pushed past .2 and deleted.
        let content_1 = fs::read_to_string(rotated_path(&log_path, 1)).unwrap();
        let content_2 = fs::read_to_string(rotated_path(&log_path, 2)).unwrap();

        assert_eq!(content_1.trim(), "third");
        assert_eq!(content_2.trim(), "second");

        // .3 should not exist since max_files=2.
        assert!(!rotated_path(&log_path, 3).exists());
    }

    #[test]
    fn test_rotated_path_format() {
        let base = PathBuf::from("/tmp/logs/metrics.jsonl");
        assert_eq!(
            rotated_path(&base, 1),
            PathBuf::from("/tmp/logs/metrics.jsonl.1")
        );
        assert_eq!(
            rotated_path(&base, 5),
            PathBuf::from("/tmp/logs/metrics.jsonl.5")
        );
    }

    #[test]
    fn test_append_after_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path(), 1, 3);
        let log_path = PathBuf::from(&config.path);

        {
            let mut logger = MetricsLogger::new(&config).unwrap();
            logger.write_line("before").unwrap();
        }

        {
            let mut logger = MetricsLogger::new(&config).unwrap();
            logger.write_line("after").unwrap();
        }

        let content = fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "before");
        assert_eq!(lines[1], "after");
    }
}
