//! Eviction policy: pure decisions over a snapshot of what is resident.

use std::fmt;
use std::time::Instant;

/// A resident model as the policy sees it.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub name: String,
    pub pinned: bool,
    pub in_flight: usize,
    pub idle_since: Instant,
    pub footprint_bytes: Option<u64>,
}

impl Candidate {
    const fn blocker(&self) -> Blocker {
        if self.pinned {
            Blocker::Pinned
        } else if self.in_flight > 0 {
            Blocker::InUse(self.in_flight)
        } else {
            Blocker::Evictable
        }
    }
}

/// Why a resident model could not be evicted to make room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocker {
    Pinned,
    InUse(usize),
    /// Idle and unpinned: evicting it would not have been enough.
    Evictable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolLimits {
    pub max_loaded_models: Option<usize>,
    /// Ceiling on MLX active memory.
    pub memory_ceiling_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSummary {
    pub name: String,
    pub footprint_bytes: Option<u64>,
    pub blocker: Blocker,
}

/// A load that was refused because it does not fit and nothing more can be evicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityRefusal {
    pub model: String,
    pub model_bytes: Option<u64>,
    pub in_use_bytes: u64,
    pub limits: PoolLimits,
    pub over_memory: bool,
    pub over_count: bool,
    /// The model's estimated size alone is over the ceiling, so evicting cannot help.
    pub model_alone_exceeds_ceiling: bool,
    pub resident: Vec<ResidentSummary>,
}

impl fmt::Display for CapacityRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = self.model_bytes.map_or_else(
            || "unknown size".to_owned(),
            |bytes| format!("about {}", format_gib(bytes)),
        );
        write!(formatter, "cannot load model '{}' ({size}): ", self.model)?;
        let mut reasons = Vec::new();
        if let (true, Some(ceiling)) = (self.over_memory, self.limits.memory_ceiling_bytes) {
            if self.model_alone_exceeds_ceiling {
                reasons.push(format!(
                    "it alone exceeds the memory ceiling of {}",
                    format_gib(ceiling)
                ));
            } else {
                reasons.push(format!(
                    "the memory ceiling is {} and {} is in use",
                    format_gib(ceiling),
                    format_gib(self.in_use_bytes)
                ));
            }
        }
        if let (true, Some(cap)) = (self.over_count, self.limits.max_loaded_models) {
            reasons.push(format!("max_loaded_models is {cap}"));
        }
        write!(formatter, "{}. ", reasons.join(" and "))?;
        if self.resident.is_empty() {
            return write!(formatter, "No other model is loaded.");
        }
        write!(formatter, "Currently loaded: ")?;
        for (index, summary) in self.resident.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            let why = match summary.blocker {
                Blocker::Pinned => "pinned, cannot be evicted".to_owned(),
                Blocker::InUse(count) => {
                    format!("in use by {count} request(s), cannot be evicted")
                }
                Blocker::Evictable => "idle, but evicting it would not free enough".to_owned(),
            };
            let footprint = summary
                .footprint_bytes
                .map_or_else(String::new, |bytes| format!(", {}", format_gib(bytes)));
            write!(formatter, "'{}' ({why}{footprint})", summary.name)?;
        }
        write!(formatter, ".")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Fits,
    /// Evict this model, then decide again.
    Evict(String),
    Refuse(Box<CapacityRefusal>),
}

/// What the policy needs to know about the load it is making room for.
#[derive(Debug, Clone, Copy)]
pub struct Incoming<'a> {
    pub name: &'a str,
    /// Bytes expected to be added on top of `in_use_bytes`; zero once the model is loaded.
    pub reserve_bytes: u64,
    /// Size reported in a refusal.
    pub model_bytes: Option<u64>,
}

/// Decide whether `incoming` fits next to `resident`, and if not, which model to evict.
///
/// The victim is the least recently used model that is idle and unpinned.
pub fn decide(
    limits: &PoolLimits,
    incoming: Incoming<'_>,
    in_use_bytes: u64,
    resident: &[Candidate],
) -> Decision {
    let over_count = limits
        .max_loaded_models
        .is_some_and(|cap| resident.len().saturating_add(1) > cap);
    let over_memory = limits
        .memory_ceiling_bytes
        .is_some_and(|ceiling| in_use_bytes.saturating_add(incoming.reserve_bytes) > ceiling);
    if !over_count && !over_memory {
        return Decision::Fits;
    }

    let alone_exceeds = limits
        .memory_ceiling_bytes
        .is_some_and(|ceiling| incoming.reserve_bytes > ceiling);
    if !alone_exceeds
        && let Some(victim) = resident
            .iter()
            .filter(|candidate| candidate.blocker() == Blocker::Evictable)
            .min_by(|left, right| {
                left.idle_since
                    .cmp(&right.idle_since)
                    .then_with(|| left.name.cmp(&right.name))
            })
    {
        return Decision::Evict(victim.name.clone());
    }

    let mut summaries: Vec<ResidentSummary> = resident
        .iter()
        .map(|candidate| ResidentSummary {
            name: candidate.name.clone(),
            footprint_bytes: candidate.footprint_bytes,
            blocker: candidate.blocker(),
        })
        .collect();
    summaries.sort_by(|left, right| left.name.cmp(&right.name));
    Decision::Refuse(Box::new(CapacityRefusal {
        model: incoming.name.to_owned(),
        model_bytes: incoming.model_bytes,
        in_use_bytes,
        limits: *limits,
        over_memory,
        over_count,
        model_alone_exceeds_ceiling: alone_exceeds,
        resident: summaries,
    }))
}

pub fn format_gib(bytes: u64) -> String {
    let hundredths = bytes.saturating_mul(100) >> 30;
    format!("{}.{:02} GiB", hundredths / 100, hundredths % 100)
}
