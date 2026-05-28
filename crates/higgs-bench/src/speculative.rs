//! Shared helpers for speculative decoding benchmarks.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One speculative benchmark mode and its environment overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialSpec {
    pub label: String,
    pub env: BTreeMap<String, String>,
}

/// Derives the request model name from either a cache snapshot path or an
/// ordinary model path/repo ID.
#[must_use]
pub fn derive_model_name(model_path: &str) -> String {
    if looks_like_huggingface_repo_id(model_path) {
        return model_path.to_owned();
    }

    let path = Path::new(model_path.trim_end_matches('/'));
    let parts: Vec<_> = path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect();

    if let Some(snapshot_idx) = parts.iter().position(|part| part == "snapshots") {
        if snapshot_idx > 0 {
            if let Some(cache_name) = parts.get(snapshot_idx - 1) {
                if let Some(cache_repo) = cache_name.strip_prefix("models--") {
                    let model_name = cache_repo.replace("--", "/");
                    if !model_name.is_empty() {
                        return model_name;
                    }
                }
            }
        }
    }

    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map_or_else(|| model_path.to_owned(), ToOwned::to_owned)
}

fn looks_like_huggingface_repo_id(model_path: &str) -> bool {
    if model_path.starts_with('/')
        || model_path.starts_with("./")
        || model_path.starts_with("../")
        || model_path.starts_with("~/")
        || model_path.contains('\\')
    {
        return false;
    }
    let mut parts = model_path.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    if parts.next().is_some() || owner.is_empty() || name.is_empty() {
        return false;
    }

    !matches!(
        owner,
        "model" | "models" | "checkpoint" | "checkpoints" | "cache" | "data" | "target"
    )
}

/// Parses a comma-separated trial list.
///
/// Supported items are `baseline`, `mtp_default`, numeric MTP draft depths,
/// `prompt_lookup`, and `prompt_lookup_unchecked`.
pub fn parse_trial_specs(input: &str) -> Result<Vec<TrialSpec>> {
    let mut trials = Vec::new();
    for raw in input.split(',') {
        let trial = raw.trim();
        if trial.is_empty() {
            continue;
        }
        trials.push(parse_trial_spec(trial).with_context(|| format!("parse trial '{trial}'"))?);
    }

    if trials.is_empty() {
        anyhow::bail!("at least one trial must be specified");
    }

    Ok(trials)
}

fn parse_trial_spec(trial: &str) -> Result<TrialSpec> {
    match trial {
        "baseline" => Ok(trial_spec("baseline_mtp_off", [("HIGGS_MTP", "0")])),
        "mtp_default" | "default" => Ok(trial_spec("mtp_default", [("HIGGS_MTP", "1")])),
        "prompt_lookup" | "plookup" => Ok(trial_spec(
            "prompt_lookup",
            [("HIGGS_MTP", "0"), ("HIGGS_PROMPT_LOOKUP", "1")],
        )),
        "prompt_lookup_unchecked" | "plookup_unchecked" => Ok(trial_spec(
            "prompt_lookup_unchecked",
            [
                ("HIGGS_MTP", "0"),
                ("HIGGS_PROMPT_LOOKUP", "1"),
                ("HIGGS_PROMPT_LOOKUP_UNCHECKED", "1"),
            ],
        )),
        raw_depth => {
            let depth = raw_depth
                .parse::<usize>()
                .with_context(|| "expected a known trial name or numeric draft depth")?;
            if depth == 0 {
                anyhow::bail!("MTP draft depth must be >= 1");
            }
            Ok(trial_spec(
                format!("mtp_draft_{depth}"),
                [
                    ("HIGGS_MTP", "1".to_owned()),
                    ("HIGGS_MTP_DRAFT_N_MAX", depth.to_string()),
                ],
            ))
        }
    }
}

fn trial_spec<K, V, I>(label: impl Into<String>, env: I) -> TrialSpec
where
    K: Into<String>,
    V: Into<String>,
    I: IntoIterator<Item = (K, V)>,
{
    TrialSpec {
        label: label.into(),
        env: env
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{TrialSpec, derive_model_name, parse_trial_specs};

    fn trial_at(trials: &[TrialSpec], index: usize) -> anyhow::Result<&TrialSpec> {
        trials
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("missing trial at index {index}"))
    }

    #[test]
    fn derive_model_name_from_huggingface_snapshot_path() {
        let model =
            derive_model_name("/cache/hub/models--org--Qwen3.6-27B-mtp/snapshots/abcdef123456");

        assert_eq!(model, "org/Qwen3.6-27B-mtp");
    }

    #[test]
    fn derive_model_name_keeps_huggingface_repo_id() {
        let model = derive_model_name("org/Qwen3.6-27B-mtp");

        assert_eq!(model, "org/Qwen3.6-27B-mtp");
    }

    #[test]
    fn derive_model_name_uses_basename_for_common_relative_model_dirs() {
        let model = derive_model_name("models/local-qwen");

        assert_eq!(model, "local-qwen");
    }

    #[test]
    fn parse_trial_specs_sets_expected_env_overrides() -> anyhow::Result<()> {
        let trials =
            parse_trial_specs("baseline,mtp_default,2,prompt_lookup,prompt_lookup_unchecked")?;

        let baseline = trial_at(&trials, 0)?;
        let mtp_default = trial_at(&trials, 1)?;
        let mtp_draft_2 = trial_at(&trials, 2)?;
        let prompt_lookup = trial_at(&trials, 3)?;
        let prompt_lookup_unchecked = trial_at(&trials, 4)?;

        assert_eq!(baseline.label, "baseline_mtp_off");
        assert_eq!(baseline.env.get("HIGGS_MTP").map(String::as_str), Some("0"));

        assert_eq!(mtp_default.label, "mtp_default");
        assert_eq!(
            mtp_default.env.get("HIGGS_MTP").map(String::as_str),
            Some("1")
        );

        assert_eq!(mtp_draft_2.label, "mtp_draft_2");
        assert_eq!(
            mtp_draft_2.env.get("HIGGS_MTP").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            mtp_draft_2
                .env
                .get("HIGGS_MTP_DRAFT_N_MAX")
                .map(String::as_str),
            Some("2")
        );

        assert_eq!(prompt_lookup.label, "prompt_lookup");
        assert_eq!(
            prompt_lookup.env.get("HIGGS_MTP").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            prompt_lookup
                .env
                .get("HIGGS_PROMPT_LOOKUP")
                .map(String::as_str),
            Some("1")
        );

        assert_eq!(prompt_lookup_unchecked.label, "prompt_lookup_unchecked");
        assert_eq!(
            prompt_lookup_unchecked
                .env
                .get("HIGGS_PROMPT_LOOKUP_UNCHECKED")
                .map(String::as_str),
            Some("1")
        );

        Ok(())
    }
}
