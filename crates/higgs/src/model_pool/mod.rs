//! On-demand model lifecycle: single-flight loading, in-flight tracking, idle expiry and
//! least-recently-used eviction under a model-count cap and a memory ceiling.
//!
//! The pool is generic over the model type and talks to the outside world only through
//! [`ModelLoader`], so the state machine and the eviction policy run in tests without a GPU.

mod policy;
mod status;

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests;

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use chrono::Utc;
use tokio::sync::Notify;
use tracing::{info, warn};

use crate::config::LoadMode;

pub use policy::{Blocker, CapacityRefusal, PoolLimits, ResidentSummary};
pub use status::{ModelState, ModelStatus};

use policy::{Candidate, Decision, Incoming};

/// Everything the pool needs from the outside world. All methods block.
pub trait ModelLoader<M>: Send + Sync + 'static {
    /// Build the model named `name`.
    fn load(&self, name: &str) -> Result<M, String>;

    /// Drop `model` and return only once its memory has been released.
    fn unload(&self, name: &str, model: Arc<M>);

    /// Footprint predicted from the model's files, if they can be found.
    fn estimate_bytes(&self, name: &str) -> Option<u64>;

    /// Memory held by live models right now, used to check the ceiling.
    fn memory_in_use(&self) -> Option<u64>;

    /// Return allocator-cached memory to the system.
    fn release_cached_memory(&self);
}

/// Loader for a pool whose models were built by the caller: they cannot be rebuilt.
pub struct ResidentOnly;

impl<M: Send + Sync + 'static> ModelLoader<M> for ResidentOnly {
    fn load(&self, name: &str) -> Result<M, String> {
        Err(format!(
            "model '{name}' was supplied already loaded and cannot be reloaded"
        ))
    }

    fn unload(&self, _name: &str, model: Arc<M>) {
        drop(model);
    }

    fn estimate_bytes(&self, _name: &str) -> Option<u64> {
        None
    }

    fn memory_in_use(&self) -> Option<u64> {
        None
    }

    fn release_cached_memory(&self) {}
}

/// Why a request for a model could not be served.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    #[error("model '{0}' is not a configured local model")]
    Unknown(String),
    #[error("model '{model}' failed to load: {reason}")]
    LoadFailed { model: String, reason: String },
    #[error("{0}")]
    OverCapacity(Box<CapacityRefusal>),
    #[error(
        "model '{model}' has {in_flight} request(s) in flight; retry when it is idle or pass force=true to wait for them"
    )]
    InUse { model: String, in_flight: usize },
    #[error("model '{model}' is {state}")]
    Busy { model: String, state: ModelState },
}

/// Per-model settings the pool acts on.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub engine_kind: &'static str,
    pub load: LoadMode,
    /// Never evicted or expired by idle time. An explicit unload still works.
    pub pinned: bool,
    pub idle_ttl: Option<Duration>,
}

pub type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// A pool of lazily loaded models of type `M`.
pub struct ModelPool<M> {
    slots: HashMap<String, Arc<Slot<M>>>,
    order: Vec<String>,
    loader: Arc<dyn ModelLoader<M>>,
    limits: PoolLimits,
    clock: Clock,
    /// Serializes load and eviction decisions so two loads cannot both count the same free
    /// memory, and so a load's measured memory growth is not another load's.
    admission: tokio::sync::Mutex<()>,
}

struct Slot<M> {
    spec: ModelSpec,
    inner: Mutex<SlotInner<M>>,
    /// Fires on every phase change and every release of a lease.
    changed: Notify,
    clock: Clock,
}

struct SlotInner<M> {
    phase: Phase<M>,
    in_flight: usize,
    last_used: Option<Instant>,
    estimated_bytes: Option<u64>,
    loads_started: u64,
}

enum Phase<M> {
    Unloaded,
    Loading {
        generation: u64,
    },
    Loaded(Resident<M>),
    /// The model has been taken out; leases may still be draining.
    Unloading,
    Failed {
        generation: u64,
        error: ModelError,
    },
}

struct Resident<M> {
    model: Arc<M>,
    loaded_at: Instant,
    load_duration: Duration,
    measured_bytes: Option<u64>,
}

enum Step<M> {
    Ready(ModelLease<M>),
    Failed(ModelError),
    Start(u64),
    Wait(Option<u64>),
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[allow(clippy::significant_drop_tightening)]
impl<M: Send + Sync + 'static> Slot<M> {
    fn new(spec: ModelSpec, clock: Clock) -> Self {
        Self {
            spec,
            inner: Mutex::new(SlotInner {
                phase: Phase::Unloaded,
                in_flight: 0,
                last_used: None,
                estimated_bytes: None,
                loads_started: 0,
            }),
            changed: Notify::new(),
            clock,
        }
    }

    const fn state(inner: &SlotInner<M>) -> ModelState {
        match inner.phase {
            Phase::Unloaded => ModelState::Unloaded,
            Phase::Loading { .. } => ModelState::Loading,
            Phase::Loaded(_) => ModelState::Loaded,
            Phase::Unloading => ModelState::Unloading,
            Phase::Failed { .. } => ModelState::Failed,
        }
    }

    /// Take the next step towards handing out a lease, deciding under one lock acquisition.
    fn step(self: &Arc<Self>, waiting_on: Option<u64>) -> Step<M> {
        let mut inner = lock(&self.inner);
        let failure = match &inner.phase {
            Phase::Loaded(resident) => {
                let model = Arc::clone(&resident.model);
                inner.in_flight += 1;
                inner.last_used = Some((self.clock)());
                return Step::Ready(ModelLease {
                    model,
                    _release: ReleaseOnDrop(Arc::clone(self)),
                });
            }
            Phase::Loading { generation } => return Step::Wait(Some(*generation)),
            Phase::Unloading => return Step::Wait(None),
            Phase::Unloaded => None,
            Phase::Failed { generation, error } => Some((*generation, error.clone())),
        };
        if let Some((generation, error)) = failure
            && waiting_on.is_some_and(|waited| generation >= waited)
        {
            return Step::Failed(error);
        }
        inner.loads_started += 1;
        let generation = inner.loads_started;
        inner.phase = Phase::Loading { generation };
        Step::Start(generation)
    }

    /// Take the model out for an automatic unload if it is idle and unpinned.
    fn begin_automatic_unload(&self) -> Option<Arc<M>> {
        let mut inner = lock(&self.inner);
        if self.spec.pinned || inner.in_flight > 0 || !matches!(inner.phase, Phase::Loaded(_)) {
            return None;
        }
        Self::take_model(&mut inner)
    }

    /// Take the model out if its idle time has reached the configured TTL.
    fn begin_expiry(&self, now: Instant) -> Option<Arc<M>> {
        let ttl = self.spec.idle_ttl?;
        let mut inner = lock(&self.inner);
        let Phase::Loaded(resident) = &inner.phase else {
            return None;
        };
        let idle_since = resident.idle_since(inner.last_used);
        let expired = now.saturating_duration_since(idle_since) >= ttl;
        if self.spec.pinned || inner.in_flight > 0 || !expired {
            return None;
        }
        Self::take_model(&mut inner)
    }

    fn begin_explicit_unload(&self, force: bool) -> Result<Option<Arc<M>>, ModelError> {
        let mut inner = lock(&self.inner);
        match inner.phase {
            Phase::Loaded(_) => {
                if inner.in_flight > 0 && !force {
                    return Err(ModelError::InUse {
                        model: self.spec.name.clone(),
                        in_flight: inner.in_flight,
                    });
                }
                Ok(Self::take_model(&mut inner))
            }
            Phase::Unloaded | Phase::Failed { .. } => Ok(None),
            Phase::Loading { .. } | Phase::Unloading => Err(ModelError::Busy {
                model: self.spec.name.clone(),
                state: Self::state(&inner),
            }),
        }
    }

    fn take_model(inner: &mut SlotInner<M>) -> Option<Arc<M>> {
        match std::mem::replace(&mut inner.phase, Phase::Unloading) {
            Phase::Loaded(resident) => Some(resident.model),
            restored @ (Phase::Unloaded
            | Phase::Loading { .. }
            | Phase::Unloading
            | Phase::Failed { .. }) => {
                inner.phase = restored;
                None
            }
        }
    }

    fn set_phase(&self, phase: Phase<M>) {
        lock(&self.inner).phase = phase;
        self.changed.notify_waiters();
    }

    async fn wait_until_idle(&self) {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if lock(&self.inner).in_flight == 0 {
                return;
            }
            notified.await;
        }
    }

    fn candidate(&self) -> Option<Candidate> {
        let inner = lock(&self.inner);
        let resident = inner.phase.resident()?;
        Some(Candidate {
            name: self.spec.name.clone(),
            pinned: self.spec.pinned,
            in_flight: inner.in_flight,
            idle_since: resident.idle_since(inner.last_used),
            footprint_bytes: resident.measured_bytes.or(inner.estimated_bytes),
        })
    }

    fn status(&self) -> ModelStatus {
        let inner = lock(&self.inner);
        let now = (self.clock)();
        let wall_clock_now = Utc::now();
        let last_used_at = inner.last_used.and_then(|used| {
            let age = chrono::Duration::from_std(now.saturating_duration_since(used)).ok()?;
            wall_clock_now.checked_sub_signed(age)
        });
        let load_duration = inner.phase.resident().map(|loaded| loaded.load_duration);
        let measured_bytes = inner
            .phase
            .resident()
            .and_then(|loaded| loaded.measured_bytes);
        let last_error = inner.phase.failure().map(ToString::to_string);
        ModelStatus {
            name: self.spec.name.clone(),
            state: Self::state(&inner),
            last_error,
            pinned: self.spec.pinned,
            load: self.spec.load,
            in_flight: inner.in_flight,
            last_used_at,
            load_duration_ms: load_duration
                .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)),
            estimated_bytes: inner.estimated_bytes,
            measured_bytes,
            engine: self.spec.engine_kind,
            idle_ttl_seconds: self.spec.idle_ttl.map(|ttl| ttl.as_secs()),
        }
    }
}

impl<M> Phase<M> {
    const fn resident(&self) -> Option<&Resident<M>> {
        if let Self::Loaded(resident) = self {
            Some(resident)
        } else {
            None
        }
    }

    const fn failure(&self) -> Option<&ModelError> {
        if let Self::Failed { error, .. } = self {
            Some(error)
        } else {
            None
        }
    }
}

impl<M> Resident<M> {
    fn idle_since(&self, last_used: Option<Instant>) -> Instant {
        last_used.map_or(self.loaded_at, |used| used.max(self.loaded_at))
    }
}

/// Use of a loaded model by one request. Derefs to the model.
///
/// The pool will not unload or evict the model while any lease is alive, and an explicit
/// forced unload waits for leases to drop, so hold it for as long as the model is in use,
/// including for the whole of a stream.
pub struct ModelLease<M> {
    model: Arc<M>,
    /// Declared after `model` so the model reference is released first: an unload that sees
    /// zero requests in flight then holds the only reference.
    _release: ReleaseOnDrop<M>,
}

struct ReleaseOnDrop<M>(Arc<Slot<M>>);

impl<M> Deref for ModelLease<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

impl<M> Drop for ReleaseOnDrop<M> {
    fn drop(&mut self) {
        {
            let mut inner = lock(&self.0.inner);
            inner.in_flight = inner.in_flight.saturating_sub(1);
            inner.last_used = Some((self.0.clock)());
        }
        self.0.changed.notify_waiters();
    }
}

impl<M: Send + Sync + 'static> ModelPool<M> {
    pub fn new(
        specs: Vec<ModelSpec>,
        loader: Arc<dyn ModelLoader<M>>,
        limits: PoolLimits,
    ) -> Arc<Self> {
        Self::with_clock(specs, loader, limits, Arc::new(Instant::now))
    }

    pub fn with_clock(
        specs: Vec<ModelSpec>,
        loader: Arc<dyn ModelLoader<M>>,
        limits: PoolLimits,
        clock: Clock,
    ) -> Arc<Self> {
        let order = specs.iter().map(|spec| spec.name.clone()).collect();
        let slots = specs
            .into_iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    Arc::new(Slot::new(spec, Arc::clone(&clock))),
                )
            })
            .collect();
        Arc::new(Self {
            slots,
            order,
            loader,
            limits,
            clock,
            admission: tokio::sync::Mutex::new(()),
        })
    }

    /// A pool over models the caller already built, with no limits. Once one is unloaded it
    /// cannot come back, so callers should pin them.
    pub fn from_resident(models: Vec<(ModelSpec, Arc<M>)>) -> Arc<Self> {
        let (specs, built): (Vec<ModelSpec>, Vec<Arc<M>>) = models.into_iter().unzip();
        let pool = Self::new(specs, Arc::new(ResidentOnly), PoolLimits::default());
        for (spec, model) in pool.order.iter().zip(built) {
            if let Some(slot) = pool.slots.get(spec) {
                slot.set_phase(Phase::Loaded(Resident {
                    model,
                    loaded_at: (pool.clock)(),
                    load_duration: Duration::ZERO,
                    measured_bytes: None,
                }));
            }
        }
        pool
    }

    pub fn contains(&self, name: &str) -> bool {
        self.slots.contains_key(name)
    }

    /// Configured model names in configuration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }

    /// A loaded model if there is one, otherwise the first configured.
    pub fn preferred_name(&self) -> Option<&str> {
        self.names()
            .find(|name| {
                self.status(name)
                    .is_some_and(|s| s.state == ModelState::Loaded)
            })
            .or_else(|| self.names().next())
    }

    pub fn status(&self, name: &str) -> Option<ModelStatus> {
        self.slots.get(name).map(|slot| slot.status())
    }

    /// Status of every configured model, sorted by name.
    pub fn statuses(&self) -> Vec<ModelStatus> {
        let mut statuses: Vec<ModelStatus> =
            self.slots.values().map(|slot| slot.status()).collect();
        statuses.sort_by(|left, right| left.name.cmp(&right.name));
        statuses
    }

    /// Get the model, loading it first if needed.
    ///
    /// Concurrent callers for a model that is not loaded share one load. If it fails, every
    /// caller waiting on it gets the error and the next call starts a fresh attempt.
    pub async fn acquire(self: &Arc<Self>, name: &str) -> Result<ModelLease<M>, ModelError> {
        let slot = self.slot(name)?;
        let mut waiting_on = None;
        loop {
            // Registered before the state is read so a change in between is not missed.
            let notified = slot.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match slot.step(waiting_on) {
                Step::Ready(lease) => return Ok(lease),
                Step::Failed(error) => return Err(error),
                Step::Start(generation) => {
                    waiting_on = Some(generation);
                    // Detached so a client that disconnects mid-load cannot strand the slot
                    // in the loading phase.
                    tokio::spawn(Arc::clone(self).run_load(Arc::clone(&slot), generation));
                }
                Step::Wait(generation) => waiting_on = generation.or(waiting_on),
            }
            notified.await;
        }
    }

    /// Load the model if needed and report its status.
    pub async fn load(self: &Arc<Self>, name: &str) -> Result<ModelStatus, ModelError> {
        drop(self.acquire(name).await?);
        self.status(name)
            .ok_or_else(|| ModelError::Unknown(name.to_owned()))
    }

    /// Load every model configured to load at startup, in configuration order.
    pub async fn load_startup_models(self: &Arc<Self>) -> Result<(), ModelError> {
        for name in &self.order {
            let loads_at_startup = self
                .slots
                .get(name)
                .is_some_and(|slot| slot.spec.load == LoadMode::Startup);
            if loads_at_startup {
                drop(self.acquire(name).await?);
            }
        }
        Ok(())
    }

    /// Unload a model. Without `force`, refuses while requests are in flight; with it, stops
    /// handing out the model and waits for those requests to finish before freeing it.
    pub async fn unload(
        self: &Arc<Self>,
        name: &str,
        force: bool,
    ) -> Result<ModelStatus, ModelError> {
        let slot = self.slot(name)?;
        if let Some(model) = slot.begin_explicit_unload(force)? {
            let pool = Arc::clone(self);
            let draining = Arc::clone(&slot);
            // Detached so a dropped HTTP request cannot leave the slot half unloaded.
            let finished = tokio::spawn(async move {
                draining.wait_until_idle().await;
                pool.finish_unload(&draining, model).await;
            })
            .await;
            if finished.is_err() {
                warn!(model = name, "unload task did not complete");
            }
        }
        Ok(slot.status())
    }

    /// Unload every model whose idle time has reached its TTL. Returns their names.
    pub async fn sweep_idle(self: &Arc<Self>) -> Vec<String> {
        let now = (self.clock)();
        let mut expired = Vec::new();
        for name in &self.order {
            let Some(slot) = self.slots.get(name) else {
                continue;
            };
            if let Some(model) = slot.begin_expiry(now) {
                info!(model = %name, "unloading model after idle timeout");
                self.finish_unload(slot, model).await;
                expired.push(name.clone());
            }
        }
        expired
    }

    /// Start the background task that applies idle TTLs, if any model has one.
    pub fn spawn_idle_sweeper(self: &Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        let shortest_ttl = self
            .slots
            .values()
            .filter(|slot| !slot.spec.pinned)
            .filter_map(|slot| slot.spec.idle_ttl)
            .min()?;
        let period = (shortest_ttl / 4).clamp(Duration::from_secs(1), Duration::from_secs(30));
        let pool = Arc::downgrade(self);
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                let Some(live_pool) = pool.upgrade() else {
                    break;
                };
                live_pool.sweep_idle().await;
            }
        }))
    }

    fn slot(&self, name: &str) -> Result<Arc<Slot<M>>, ModelError> {
        self.slots
            .get(name)
            .map(Arc::clone)
            .ok_or_else(|| ModelError::Unknown(name.to_owned()))
    }

    async fn run_load(self: Arc<Self>, slot: Arc<Slot<M>>, generation: u64) {
        let phase = match self.load_resident(&slot).await {
            Ok(resident) => Phase::Loaded(resident),
            Err(error) => {
                warn!(model = %slot.spec.name, %error, "model load failed");
                Phase::Failed { generation, error }
            }
        };
        slot.set_phase(phase);
    }

    async fn load_resident(&self, slot: &Arc<Slot<M>>) -> Result<Resident<M>, ModelError> {
        let name = slot.spec.name.as_str();
        let _admission = self.admission.lock().await;

        let estimate = self.estimate_bytes(slot).await;
        self.make_room(slot, estimate.unwrap_or(0), estimate)
            .await?;

        let memory_before = self.loader.memory_in_use();
        let started = Instant::now();
        let loader = Arc::clone(&self.loader);
        let load_name = name.to_owned();
        let built = tokio::task::spawn_blocking(move || loader.load(&load_name))
            .await
            .map_err(|_| ModelError::LoadFailed {
                model: name.to_owned(),
                reason: "the load task panicked".to_owned(),
            })?
            .map_err(|reason| ModelError::LoadFailed {
                model: name.to_owned(),
                reason,
            })?;
        let load_duration = started.elapsed();
        let measured_bytes = match (memory_before, self.loader.memory_in_use()) {
            (Some(before), Some(after)) => Some(after.saturating_sub(before)),
            _ => None,
        };
        info!(
            model = name,
            load_ms = u64::try_from(load_duration.as_millis()).unwrap_or(u64::MAX),
            measured_bytes,
            estimated_bytes = estimate,
            "model loaded"
        );

        let model = Arc::new(built);
        // The estimate may have been low or missing; check the ceiling again with the real
        // figure, now that the new model's memory is counted in use.
        if let Err(error) = self.make_room(slot, 0, measured_bytes.or(estimate)).await {
            self.release_unregistered(name, model).await;
            return Err(error);
        }
        Ok(Resident {
            model,
            loaded_at: (self.clock)(),
            load_duration,
            measured_bytes,
        })
    }

    async fn estimate_bytes(&self, slot: &Arc<Slot<M>>) -> Option<u64> {
        let cached = lock(&slot.inner).estimated_bytes;
        if cached.is_some() {
            return cached;
        }
        let loader = Arc::clone(&self.loader);
        let name = slot.spec.name.clone();
        let estimate = tokio::task::spawn_blocking(move || loader.estimate_bytes(&name))
            .await
            .ok()
            .flatten();
        lock(&slot.inner).estimated_bytes = estimate;
        estimate
    }

    /// Evict least-recently-used idle models until `slot`'s model fits, or refuse.
    async fn make_room(
        &self,
        slot: &Arc<Slot<M>>,
        reserve_bytes: u64,
        model_bytes: Option<u64>,
    ) -> Result<(), ModelError> {
        // Each pass evicts one model, so more passes than slots means eviction is not progressing.
        for _ in 0..=self.slots.len() {
            let resident: Vec<Candidate> = self
                .slots
                .values()
                .filter(|other| !Arc::ptr_eq(other, slot))
                .filter_map(|other| other.candidate())
                .collect();
            let in_use_bytes = self.loader.memory_in_use().unwrap_or_else(|| {
                resident
                    .iter()
                    .filter_map(|candidate| candidate.footprint_bytes)
                    .fold(0, u64::saturating_add)
            });
            let incoming = Incoming {
                name: &slot.spec.name,
                reserve_bytes,
                model_bytes,
            };
            match policy::decide(&self.limits, incoming, in_use_bytes, &resident) {
                Decision::Fits => return Ok(()),
                Decision::Evict(victim) => {
                    info!(
                        victim = %victim,
                        for_model = %slot.spec.name,
                        "evicting least recently used model to make room"
                    );
                    self.evict(&victim).await;
                }
                Decision::Refuse(refusal) => return Err(ModelError::OverCapacity(refusal)),
            }
        }
        Err(ModelError::LoadFailed {
            model: slot.spec.name.clone(),
            reason: "could not evict enough models to make room".to_owned(),
        })
    }

    async fn evict(&self, name: &str) {
        let Some(slot) = self.slots.get(name) else {
            return;
        };
        if let Some(model) = slot.begin_automatic_unload() {
            self.finish_unload(slot, model).await;
        }
    }

    /// Free `model` on a blocking thread, then mark the slot unloaded.
    async fn finish_unload(&self, slot: &Arc<Slot<M>>, model: Arc<M>) {
        self.release_unregistered(&slot.spec.name, model).await;
        slot.set_phase(Phase::Unloaded);
    }

    async fn release_unregistered(&self, name: &str, model: Arc<M>) {
        let loader = Arc::clone(&self.loader);
        let unload_name = name.to_owned();
        let released = tokio::task::spawn_blocking(move || {
            loader.unload(&unload_name, model);
            loader.release_cached_memory();
        })
        .await;
        if released.is_err() {
            warn!(model = name, "unloading the model panicked");
        }
    }
}
