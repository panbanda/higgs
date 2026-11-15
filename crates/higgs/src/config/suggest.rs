//! "Did you mean" suggestions for keys nobody defined.

/// Levenshtein distance between two strings, ignoring ASCII case.
pub fn edit_distance(first: &str, second: &str) -> usize {
    let first_chars: Vec<char> = first.chars().map(|c| c.to_ascii_lowercase()).collect();
    let second_chars: Vec<char> = second.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut previous: Vec<usize> = (0..=second_chars.len()).collect();
    for (row, left_char) in first_chars.iter().enumerate() {
        let mut current = vec![row + 1];
        for (column, right_char) in second_chars.iter().enumerate() {
            let substitution =
                previous.get(column).copied().unwrap_or(0) + usize::from(left_char != right_char);
            let insertion = current.last().copied().unwrap_or(0) + 1;
            let deletion = previous.get(column + 1).copied().unwrap_or(0) + 1;
            current.push(substitution.min(insertion).min(deletion));
        }
        previous = current;
    }
    previous.last().copied().unwrap_or(0)
}

/// A name that can be typed in a section, and what to suggest when it is the closest match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub typed: String,
    pub shown: String,
}

impl Candidate {
    pub fn same(name: &str) -> Self {
        Self {
            typed: name.to_owned(),
            shown: name.to_owned(),
        }
    }

    pub fn renamed(old: &str, new: &str) -> Self {
        Self {
            typed: old.to_owned(),
            shown: new.to_owned(),
        }
    }
}

/// The candidate closest to `name`, if any is close enough to be a plausible typo.
pub fn closest<'a>(name: &str, candidates: &'a [Candidate]) -> Option<&'a Candidate> {
    let threshold = (name.chars().count() / 3).max(2);
    candidates
        .iter()
        .map(|candidate| (edit_distance(name, &candidate.typed), candidate))
        .filter(|(distance, _)| *distance <= threshold)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

/// The message for an unknown key: the key, its section, and the closest valid key.
pub fn unknown_key_message(name: &str, section: &str, candidates: &[Candidate]) -> String {
    if let Some(candidate) = closest(name, candidates) {
        return format!(
            "unknown key `{name}` in {section}; did you mean `{}`?",
            candidate.shown
        );
    }
    let mut valid: Vec<&str> = candidates
        .iter()
        .filter(|candidate| candidate.typed == candidate.shown)
        .map(|candidate| candidate.shown.as_str())
        .collect();
    valid.sort_unstable();
    valid.dedup();
    if valid.is_empty() {
        format!("unknown key `{name}` in {section}")
    } else {
        format!(
            "unknown key `{name}` in {section}; valid keys are: {}",
            valid.join(", ")
        )
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_counts_single_edits() {
        assert_eq!(edit_distance("idle_ttl", "idle_ttl"), 0);
        assert_eq!(edit_distance("idel_ttl", "idle_ttl"), 2);
        assert_eq!(edit_distance("port", "ports"), 1);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("ABC", "abc"), 0);
    }

    #[test]
    fn the_closest_plausible_candidate_is_suggested() {
        let candidates = [Candidate::same("idle_ttl"), Candidate::same("pinned")];
        assert_eq!(closest("idel_ttl", &candidates).unwrap().shown, "idle_ttl");
        assert!(closest("completely_different", &candidates).is_none());
    }

    #[test]
    fn a_renamed_candidate_suggests_the_new_name() {
        let candidates = [Candidate::renamed("kv_bits", "kv_cache.bits")];
        let message = unknown_key_message("kv_bitz", "[[models]]", &candidates);
        assert!(
            message.contains("did you mean `kv_cache.bits`"),
            "{message}"
        );
    }

    #[test]
    fn without_a_close_match_the_valid_keys_are_listed() {
        let candidates = [Candidate::same("host"), Candidate::same("port")];
        let message = unknown_key_message("zzzzzzzz", "[server]", &candidates);
        assert_eq!(
            message,
            "unknown key `zzzzzzzz` in [server]; valid keys are: host, port"
        );
    }
}
