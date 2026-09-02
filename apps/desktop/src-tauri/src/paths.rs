//! Path containment shared by [`crate::local`] (config/log paths) and
//! [`crate::hub`] (the Hugging Face cache layout).

use std::path::{Path, PathBuf};

/// Lexically removes `.` and `..` components without touching the
/// filesystem, so containment checks hold for files that do not exist yet.
pub fn normalized(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Symlink-safe containment check: true when `candidate` lexically resolves
/// under `root` AND no existing ancestor directory between `root` and
/// `candidate` is itself a symlink, so a symlinked directory placed inside
/// `root` cannot redirect a read or write outside it.
///
/// Deliberately does not inspect `candidate` itself: callers that create or
/// replace a symlink at `candidate` (the Hugging Face cache's blob links)
/// must be able to do so even when a previous, legitimately-contained
/// symlink already sits there.
///
/// When `root` does not exist on disk yet, only the lexical check applies,
/// since there is nothing to canonicalize.
pub fn is_contained(root: &Path, candidate: &Path) -> bool {
    if !normalized(candidate).starts_with(normalized(root)) {
        return false;
    }
    let Ok(canonical_root) = std::fs::canonicalize(root) else {
        return true;
    };

    // Walk every ancestor directory from `candidate`'s parent up to `root`,
    // rejecting any that is itself a symlink. `symlink_metadata` (lstat)
    // reports the *last* path component's own symlink-ness without
    // following it, even though earlier components are still resolved
    // transparently by the OS, so this catches a symlink at any depth once
    // the walk reaches it as the final segment being inspected.
    let mut dir = candidate.parent();
    let mut nearest_existing: Option<PathBuf> = None;
    while let Some(path) = dir {
        if normalized(path) == normalized(root) {
            break;
        }
        if let Ok(meta) = std::fs::symlink_metadata(path) {
            if meta.file_type().is_symlink() {
                return false;
            }
            if nearest_existing.is_none() {
                nearest_existing = Some(path.to_path_buf());
            }
        }
        dir = path.parent();
    }

    let nearest = nearest_existing.unwrap_or_else(|| root.to_path_buf());
    std::fs::canonicalize(&nearest).is_ok_and(|resolved| resolved.starts_with(&canonical_root))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "higgs-desktop-paths-test-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn rejects_lexical_escape() {
        let root = temp_dir("lexical-root");
        assert!(!is_contained(&root, Path::new("/etc/passwd")));
        assert!(!is_contained(&root, &root.join("../escaped")));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn accepts_a_plain_nested_path() {
        let root = temp_dir("plain-root");
        std::fs::create_dir_all(root.join("a/b")).expect("nested dirs");
        assert!(is_contained(&root, &root.join("a/b/file.txt")));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn accepts_the_leaf_itself_being_a_symlink() {
        // The Hugging Face cache intentionally symlinks snapshot files to
        // blobs; the leaf being a symlink must not be rejected.
        let root = temp_dir("leaf-symlink-root");
        let blob = root.join("blob-target");
        std::fs::write(&blob, "data").expect("write blob");
        let link = root.join("snapshot-link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, &link).expect("symlink");
        #[cfg(unix)]
        assert!(is_contained(&root, &link));
        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_an_ancestor_symlink_escaping_the_root() {
        let root = temp_dir("ancestor-symlink-root");
        let outside = temp_dir("ancestor-symlink-outside");
        let linked = root.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).expect("symlink dir");
        let candidate = linked.join("escaped.txt");
        assert!(!is_contained(&root, &candidate));
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_an_ancestor_symlink_reachable_only_through_a_pre_existing_child() {
        // The symlinked ancestor's target already contains the child
        // component, so a naive "stop at the first existing ancestor" walk
        // would find `linked/child` first (through the symlink) and never
        // notice `linked` itself is a symlink out of `root`.
        let root = temp_dir("ancestor-symlink-root-2");
        let outside = temp_dir("ancestor-symlink-outside-2");
        std::fs::create_dir_all(outside.join("child")).expect("pre-existing child");
        let linked = root.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).expect("symlink dir");
        let candidate = linked.join("child").join("escaped.txt");
        assert!(!is_contained(&root, &candidate));
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&outside).ok();
    }
}
