use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::policy::{Blocker, Candidate, Decision, Incoming, decide};
use super::*;

const GIB: u64 = 1 << 30;

/// A loader whose models are just their names, with scripted sizes, delays and failures.
#[derive(Default)]
struct FakeLoader {
    actual_bytes: Mutex<HashMap<String, u64>>,
    estimated_bytes: Mutex<HashMap<String, u64>>,
    in_use_bytes: AtomicU64,
    reports_memory: bool,
    load_delay: Duration,
    load_calls: AtomicUsize,
    scripted_failures: Mutex<VecDeque<String>>,
    events: Mutex<Vec<String>>,
}

impl FakeLoader {
    fn with_sizes(sizes_gib: &[(&str, u64)]) -> Self {
        let bytes: HashMap<String, u64> = sizes_gib
            .iter()
            .map(|(name, gib)| ((*name).to_owned(), gib * GIB))
            .collect();
        Self {
            actual_bytes: Mutex::new(bytes.clone()),
            estimated_bytes: Mutex::new(bytes),
            reports_memory: true,
            ..Self::default()
        }
    }

    fn fail_next_load(&self, reason: &str) {
        self.scripted_failures
            .lock()
            .unwrap()
            .push_back(reason.to_owned());
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }

    fn loads(&self) -> usize {
        self.load_calls.load(Ordering::SeqCst)
    }
}

impl ModelLoader<String> for FakeLoader {
    fn load(&self, name: &str) -> Result<String, String> {
        self.load_calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.load_delay);
        let scripted_failure = self.scripted_failures.lock().unwrap().pop_front();
        if let Some(reason) = scripted_failure {
            return Err(reason);
        }
        let bytes = self
            .actual_bytes
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .unwrap_or(0);
        self.in_use_bytes.fetch_add(bytes, Ordering::SeqCst);
        self.events.lock().unwrap().push(format!("load:{name}"));
        Ok(name.to_owned())
    }

    fn unload(&self, name: &str, model: Arc<String>) {
        let event = if Arc::strong_count(&model) == 1 {
            format!("unload:{name}")
        } else {
            format!("unload-while-in-use:{name}")
        };
        let bytes = self
            .actual_bytes
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .unwrap_or(0);
        self.in_use_bytes.fetch_sub(bytes, Ordering::SeqCst);
        self.events.lock().unwrap().push(event);
    }

    fn estimate_bytes(&self, name: &str) -> Option<u64> {
        self.estimated_bytes.lock().unwrap().get(name).copied()
    }

    fn memory_in_use(&self) -> Option<u64> {
        self.reports_memory
            .then(|| self.in_use_bytes.load(Ordering::SeqCst))
    }

    fn release_cached_memory(&self) {
        self.events.lock().unwrap().push("release-cache".to_owned());
    }
}

#[derive(Clone)]
struct FakeClock(Arc<Mutex<Instant>>);

impl FakeClock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Instant::now())))
    }

    fn advance(&self, by: Duration) {
        *self.0.lock().unwrap() += by;
    }

    fn as_clock(&self) -> Clock {
        let shared = Arc::clone(&self.0);
        Arc::new(move || *shared.lock().unwrap())
    }
}

fn spec(name: &str) -> ModelSpec {
    ModelSpec {
        name: name.to_owned(),
        engine_kind: "simple",
        load: LoadMode::OnDemand,
        pinned: false,
        idle_ttl: None,
    }
}

fn pinned(name: &str) -> ModelSpec {
    ModelSpec {
        pinned: true,
        ..spec(name)
    }
}

fn with_ttl(name: &str, seconds: u64) -> ModelSpec {
    ModelSpec {
        idle_ttl: Some(Duration::from_secs(seconds)),
        ..spec(name)
    }
}

struct Fixture {
    pool: Arc<ModelPool<String>>,
    loader: Arc<FakeLoader>,
    clock: FakeClock,
}

fn fixture(specs: Vec<ModelSpec>, fake_loader: FakeLoader, limits: PoolLimits) -> Fixture {
    let loader = Arc::new(fake_loader);
    let shared_loader: Arc<dyn ModelLoader<String>> = Arc::<FakeLoader>::clone(&loader);
    let clock = FakeClock::new();
    let pool = ModelPool::with_clock(specs, shared_loader, limits, clock.as_clock());
    Fixture {
        pool,
        loader,
        clock,
    }
}

fn cap(max_loaded_models: usize) -> PoolLimits {
    PoolLimits {
        max_loaded_models: Some(max_loaded_models),
        memory_ceiling_bytes: None,
    }
}

fn ceiling_gib(gib: u64) -> PoolLimits {
    PoolLimits {
        max_loaded_models: None,
        memory_ceiling_bytes: Some(gib * GIB),
    }
}

async fn touch(pool: &Arc<ModelPool<String>>, name: &str) {
    drop(pool.acquire(name).await.unwrap());
}

fn state_of(fixture: &Fixture, name: &str) -> ModelState {
    fixture.pool.status(name).unwrap().state
}

// -- Loading ----------------------------------------------------------------

#[tokio::test]
async fn acquire_loads_on_first_use_only() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    assert_eq!(state_of(&fx, "a"), ModelState::Unloaded);

    let first = fx.pool.acquire("a").await.unwrap();
    let second = fx.pool.acquire("a").await.unwrap();

    assert_eq!(*first, "a");
    assert_eq!(*second, "a");
    assert_eq!(fx.loader.loads(), 1);
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);
}

#[tokio::test]
async fn unknown_model_is_reported() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );

    let error = fx.pool.acquire("missing").await.err().unwrap();

    assert!(matches!(error, ModelError::Unknown(name) if name == "missing"));
}

#[tokio::test]
async fn concurrent_first_requests_share_one_load() {
    let loader = FakeLoader {
        load_delay: Duration::from_millis(100),
        ..FakeLoader::default()
    };
    let fx = fixture(vec![spec("a")], loader, PoolLimits::default());

    let leases = futures::future::join_all((0..8).map(|_| fx.pool.acquire("a"))).await;

    assert_eq!(fx.loader.loads(), 1);
    assert!(leases.iter().all(Result::is_ok));
    assert_eq!(fx.pool.status("a").unwrap().in_flight, 8);
}

#[tokio::test]
async fn failed_load_reaches_every_waiter_once_and_is_retryable() {
    let loader = FakeLoader {
        load_delay: Duration::from_millis(100),
        ..FakeLoader::default()
    };
    loader.fail_next_load("weights are corrupt");
    let fx = fixture(vec![spec("a")], loader, PoolLimits::default());

    let results = futures::future::join_all((0..5).map(|_| fx.pool.acquire("a"))).await;

    assert_eq!(fx.loader.loads(), 1, "waiters must not each retry");
    for result in results {
        let message = result.err().unwrap().to_string();
        assert!(message.contains("weights are corrupt"), "{message}");
        assert!(message.contains("'a'"), "{message}");
    }
    let status = fx.pool.status("a").unwrap();
    assert_eq!(status.state, ModelState::Failed);
    assert!(status.last_error.unwrap().contains("weights are corrupt"));

    let lease = fx.pool.acquire("a").await.unwrap();
    assert_eq!(*lease, "a");
    assert_eq!(fx.loader.loads(), 2);
    let retried = fx.pool.status("a").unwrap();
    assert_eq!(retried.state, ModelState::Loaded);
    assert_eq!(retried.last_error, None);
}

#[tokio::test]
async fn abandoned_acquire_does_not_strand_the_load() {
    let loader = FakeLoader {
        load_delay: Duration::from_millis(150),
        ..FakeLoader::default()
    };
    let fx = fixture(vec![spec("a")], loader, PoolLimits::default());

    let abandoned = tokio::time::timeout(Duration::from_millis(10), fx.pool.acquire("a")).await;
    assert!(abandoned.is_err());

    let lease = fx.pool.acquire("a").await.unwrap();
    assert_eq!(*lease, "a");
    assert_eq!(fx.loader.loads(), 1);
}

#[tokio::test]
async fn load_startup_models_skips_on_demand_models() {
    let startup = ModelSpec {
        load: LoadMode::Startup,
        ..spec("eager")
    };
    let fx = fixture(
        vec![startup, spec("lazy")],
        FakeLoader::default(),
        PoolLimits::default(),
    );

    fx.pool.load_startup_models().await.unwrap();

    assert_eq!(state_of(&fx, "eager"), ModelState::Loaded);
    assert_eq!(state_of(&fx, "lazy"), ModelState::Unloaded);
}

#[tokio::test]
async fn load_startup_models_reports_a_failure() {
    let loader = FakeLoader::default();
    loader.fail_next_load("no such file");
    let startup = ModelSpec {
        load: LoadMode::Startup,
        ..spec("eager")
    };
    let fx = fixture(vec![startup], loader, PoolLimits::default());

    let error = fx.pool.load_startup_models().await.unwrap_err();

    assert!(error.to_string().contains("no such file"));
}

// -- Eviction ---------------------------------------------------------------

#[tokio::test]
async fn count_cap_evicts_the_least_recently_used_idle_model() {
    let fx = fixture(
        vec![spec("a"), spec("b"), spec("c")],
        FakeLoader::default(),
        cap(2),
    );
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(1));
    touch(&fx.pool, "b").await;
    fx.clock.advance(Duration::from_secs(1));
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(1));

    touch(&fx.pool, "c").await;

    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "c"), ModelState::Loaded);
    assert_eq!(
        fx.loader.events(),
        ["load:a", "load:b", "unload:b", "release-cache", "load:c"]
    );
}

#[tokio::test]
async fn capacity_of_one_swaps_models_back_and_forth() {
    let fx = fixture(vec![spec("a"), spec("b")], FakeLoader::default(), cap(1));

    touch(&fx.pool, "a").await;
    touch(&fx.pool, "b").await;
    touch(&fx.pool, "a").await;

    assert_eq!(fx.loader.loads(), 3);
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Unloaded);
}

#[tokio::test]
async fn pinned_models_are_never_evicted() {
    let fx = fixture(
        vec![pinned("keep"), spec("b"), spec("c")],
        FakeLoader::default(),
        cap(2),
    );
    touch(&fx.pool, "keep").await;
    fx.clock.advance(Duration::from_secs(1));
    touch(&fx.pool, "b").await;
    fx.clock.advance(Duration::from_secs(1));

    touch(&fx.pool, "c").await;

    assert_eq!(state_of(&fx, "keep"), ModelState::Loaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "c"), ModelState::Loaded);
}

#[tokio::test]
async fn load_is_refused_when_only_pinned_models_are_loaded() {
    let fx = fixture(
        vec![pinned("keep"), spec("b")],
        FakeLoader::with_sizes(&[("keep", 6), ("b", 6)]),
        ceiling_gib(10),
    );
    touch(&fx.pool, "keep").await;

    let error = fx.pool.acquire("b").await.err().unwrap();

    let message = error.to_string();
    assert!(matches!(error, ModelError::OverCapacity(_)));
    assert!(message.contains("'b'"), "{message}");
    assert!(message.contains("6.00 GiB"), "{message}");
    assert!(message.contains("10.00 GiB"), "{message}");
    assert!(message.contains("'keep'"), "{message}");
    assert!(message.contains("pinned"), "{message}");
    assert_eq!(state_of(&fx, "keep"), ModelState::Loaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Failed);
    assert_eq!(
        fx.loader.loads(),
        1,
        "a refused load must not run the loader"
    );
}

#[tokio::test]
async fn models_in_use_are_never_evicted() {
    let fx = fixture(vec![spec("a"), spec("b")], FakeLoader::default(), cap(1));
    let busy = fx.pool.acquire("a").await.unwrap();

    let error = fx.pool.acquire("b").await.err().unwrap();

    let message = error.to_string();
    assert!(matches!(error, ModelError::OverCapacity(_)), "{message}");
    assert!(message.contains("in use by 1 request"), "{message}");
    assert!(message.contains("max_loaded_models is 1"), "{message}");
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);

    drop(busy);
    touch(&fx.pool, "b").await;
    assert_eq!(state_of(&fx, "a"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Loaded);
}

#[tokio::test]
async fn memory_ceiling_evicts_least_recently_used_until_the_model_fits() {
    let fx = fixture(
        vec![spec("a"), spec("b"), spec("c")],
        FakeLoader::with_sizes(&[("a", 4), ("b", 4), ("c", 7)]),
        ceiling_gib(10),
    );
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(1));
    touch(&fx.pool, "b").await;
    fx.clock.advance(Duration::from_secs(1));

    touch(&fx.pool, "c").await;

    assert_eq!(state_of(&fx, "a"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "c"), ModelState::Loaded);
    assert_eq!(
        fx.loader.events(),
        [
            "load:a",
            "load:b",
            "unload:a",
            "release-cache",
            "unload:b",
            "release-cache",
            "load:c"
        ]
    );
}

#[tokio::test]
async fn memory_ceiling_keeps_models_that_still_fit() {
    let fx = fixture(
        vec![spec("a"), spec("b"), spec("c")],
        FakeLoader::with_sizes(&[("a", 4), ("b", 4), ("c", 2)]),
        ceiling_gib(10),
    );
    touch(&fx.pool, "a").await;
    touch(&fx.pool, "b").await;

    touch(&fx.pool, "c").await;

    assert_eq!(fx.loader.loads(), 3);
    assert!(!fx.loader.events().iter().any(|e| e.starts_with("unload")));
}

#[tokio::test]
async fn model_larger_than_the_ceiling_is_refused_without_evicting_anything() {
    let fx = fixture(
        vec![spec("small"), spec("huge")],
        FakeLoader::with_sizes(&[("small", 2), ("huge", 20)]),
        ceiling_gib(10),
    );
    touch(&fx.pool, "small").await;

    let error = fx.pool.acquire("huge").await.err().unwrap();

    let message = error.to_string();
    assert!(message.contains("alone exceeds"), "{message}");
    assert!(message.contains("20.00 GiB"), "{message}");
    assert!(message.contains("'small'"), "{message}");
    assert_eq!(state_of(&fx, "small"), ModelState::Loaded);
}

#[tokio::test]
async fn refusal_lists_why_each_loaded_model_could_not_be_evicted() {
    let fx = fixture(
        vec![pinned("pinned-model"), spec("busy-model"), spec("wanted")],
        FakeLoader::with_sizes(&[("pinned-model", 4), ("busy-model", 4), ("wanted", 4)]),
        ceiling_gib(10),
    );
    touch(&fx.pool, "pinned-model").await;
    let _busy = fx.pool.acquire("busy-model").await.unwrap();

    let message = fx.pool.acquire("wanted").await.err().unwrap().to_string();

    assert!(
        message.contains("'pinned-model' (pinned, cannot be evicted, 4.00 GiB)"),
        "{message}"
    );
    assert!(
        message.contains("'busy-model' (in use by 1 request(s), cannot be evicted, 4.00 GiB)"),
        "{message}"
    );
}

#[tokio::test]
async fn a_load_that_turns_out_larger_than_estimated_is_checked_after_loading() {
    let loader = FakeLoader::with_sizes(&[("a", 4), ("b", 12)]);
    loader
        .estimated_bytes
        .lock()
        .unwrap()
        .insert("b".to_owned(), GIB);
    let fx = fixture(vec![spec("a"), spec("b")], loader, ceiling_gib(10));
    touch(&fx.pool, "a").await;

    let error = fx.pool.acquire("b").await.err().unwrap();

    let message = error.to_string();
    assert!(matches!(error, ModelError::OverCapacity(_)), "{message}");
    assert!(message.contains("12.00 GiB"), "{message}");
    assert_eq!(state_of(&fx, "b"), ModelState::Failed);
    assert!(fx.loader.events().contains(&"unload:b".to_owned()));
}

#[tokio::test]
async fn a_load_larger_than_estimated_evicts_idle_models_to_fit() {
    let loader = FakeLoader::with_sizes(&[("a", 4), ("b", 8)]);
    loader
        .estimated_bytes
        .lock()
        .unwrap()
        .insert("b".to_owned(), GIB);
    let fx = fixture(vec![spec("a"), spec("b")], loader, ceiling_gib(10));
    touch(&fx.pool, "a").await;

    touch(&fx.pool, "b").await;

    assert_eq!(state_of(&fx, "a"), ModelState::Unloaded);
    assert_eq!(state_of(&fx, "b"), ModelState::Loaded);
}

#[tokio::test]
async fn unknown_estimate_loads_and_is_measured() {
    let loader = FakeLoader::with_sizes(&[("a", 3)]);
    loader.estimated_bytes.lock().unwrap().clear();
    let fx = fixture(vec![spec("a")], loader, ceiling_gib(10));

    touch(&fx.pool, "a").await;

    let status = fx.pool.status("a").unwrap();
    assert_eq!(status.estimated_bytes, None);
    assert_eq!(status.measured_bytes, Some(3 * GIB));
}

// -- Idle expiry ------------------------------------------------------------

#[tokio::test]
async fn idle_model_expires_after_its_ttl() {
    let fx = fixture(
        vec![with_ttl("a", 10)],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;

    fx.clock.advance(Duration::from_secs(9));
    assert!(fx.pool.sweep_idle().await.is_empty());
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);

    fx.clock.advance(Duration::from_secs(2));
    assert_eq!(fx.pool.sweep_idle().await, ["a"]);
    assert_eq!(state_of(&fx, "a"), ModelState::Unloaded);
    assert_eq!(fx.loader.events(), ["load:a", "unload:a", "release-cache"]);
}

#[tokio::test]
async fn model_with_requests_in_flight_never_expires() {
    let fx = fixture(
        vec![with_ttl("a", 10)],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    let lease = fx.pool.acquire("a").await.unwrap();

    fx.clock.advance(Duration::from_secs(1000));
    assert!(fx.pool.sweep_idle().await.is_empty());
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);

    drop(lease);
    assert!(
        fx.pool.sweep_idle().await.is_empty(),
        "the idle clock starts when the last request finishes"
    );
    fx.clock.advance(Duration::from_secs(11));
    assert_eq!(fx.pool.sweep_idle().await, ["a"]);
}

#[tokio::test]
async fn using_a_model_postpones_its_expiry() {
    let fx = fixture(
        vec![with_ttl("a", 10)],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(8));
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(8));

    assert!(fx.pool.sweep_idle().await.is_empty());
}

#[tokio::test]
async fn pinned_models_do_not_expire() {
    let expiring_but_pinned = ModelSpec {
        idle_ttl: Some(Duration::from_secs(1)),
        ..pinned("a")
    };
    let fx = fixture(
        vec![expiring_but_pinned],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;

    fx.clock.advance(Duration::from_secs(1000));

    assert!(fx.pool.sweep_idle().await.is_empty());
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);
}

#[tokio::test]
async fn models_without_a_ttl_do_not_expire() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;

    fx.clock.advance(Duration::from_secs(100_000));

    assert!(fx.pool.sweep_idle().await.is_empty());
}

#[tokio::test]
async fn expired_model_reloads_on_the_next_request() {
    let fx = fixture(
        vec![with_ttl("a", 10)],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;
    fx.clock.advance(Duration::from_secs(20));
    fx.pool.sweep_idle().await;

    touch(&fx.pool, "a").await;

    assert_eq!(fx.loader.loads(), 2);
}

#[tokio::test]
async fn sweeper_is_only_started_when_some_unpinned_model_has_a_ttl() {
    let without = fixture(
        vec![spec("a"), pinned("b")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    assert!(without.pool.spawn_idle_sweeper().is_none());

    let only_pinned = ModelSpec {
        idle_ttl: Some(Duration::from_secs(5)),
        ..pinned("a")
    };
    let pinned_only = fixture(
        vec![only_pinned],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    assert!(pinned_only.pool.spawn_idle_sweeper().is_none());

    let with = fixture(
        vec![with_ttl("a", 5)],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    let handle = with.pool.spawn_idle_sweeper().unwrap();
    handle.abort();
}

// -- Explicit unload --------------------------------------------------------

#[tokio::test]
async fn unload_frees_the_model_then_clears_the_cache() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::with_sizes(&[("a", 4)]),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;
    assert_eq!(fx.loader.memory_in_use(), Some(4 * GIB));

    let status = fx.pool.unload("a", false).await.unwrap();

    assert_eq!(status.state, ModelState::Unloaded);
    assert_eq!(fx.loader.memory_in_use(), Some(0));
    assert_eq!(fx.loader.events(), ["load:a", "unload:a", "release-cache"]);
}

#[tokio::test]
async fn unload_of_an_unloaded_model_is_a_no_op() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );

    let status = fx.pool.unload("a", false).await.unwrap();

    assert_eq!(status.state, ModelState::Unloaded);
    assert!(fx.loader.events().is_empty());
}

#[tokio::test]
async fn unload_is_refused_while_requests_are_in_flight() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    let _lease = fx.pool.acquire("a").await.unwrap();

    let error = fx.pool.unload("a", false).await.unwrap_err();

    assert!(
        matches!(error, ModelError::InUse { in_flight: 1, .. }),
        "{error}"
    );
    assert_eq!(state_of(&fx, "a"), ModelState::Loaded);
}

#[tokio::test]
async fn pinned_models_can_be_unloaded_explicitly() {
    let fx = fixture(
        vec![pinned("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    touch(&fx.pool, "a").await;

    let status = fx.pool.unload("a", false).await.unwrap();

    assert_eq!(status.state, ModelState::Unloaded);
}

#[tokio::test]
async fn forced_unload_waits_for_in_flight_requests_to_finish() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    let lease = fx.pool.acquire("a").await.unwrap();

    let pool = Arc::clone(&fx.pool);
    let unloading = tokio::spawn(async move { pool.unload("a", true).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(!unloading.is_finished(), "must wait for the stream to end");
    let status = fx.pool.status("a").unwrap();
    assert_eq!(status.state, ModelState::Unloading);
    assert_eq!(status.in_flight, 1);
    assert_eq!(*lease, "a", "the running request keeps its model");
    assert_eq!(fx.loader.events(), ["load:a"]);

    drop(lease);
    let finished = unloading.await.unwrap().unwrap();

    assert_eq!(finished.state, ModelState::Unloaded);
    assert_eq!(fx.loader.events(), ["load:a", "unload:a", "release-cache"]);
}

#[tokio::test]
async fn requests_arriving_during_a_forced_unload_wait_and_reload() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    let lease = fx.pool.acquire("a").await.unwrap();
    let pool = Arc::clone(&fx.pool);
    let unloading = tokio::spawn(async move { pool.unload("a", true).await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let arriving_pool = Arc::clone(&fx.pool);
    let arriving =
        tokio::spawn(async move { arriving_pool.acquire("a").await.map(|held| (*held).clone()) });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!arriving.is_finished());

    drop(lease);
    unloading.await.unwrap().unwrap();

    assert_eq!(arriving.await.unwrap().unwrap(), "a");
    assert_eq!(fx.loader.loads(), 2);
}

#[tokio::test]
async fn unload_during_a_load_is_refused() {
    let loader = FakeLoader {
        load_delay: Duration::from_millis(200),
        ..FakeLoader::default()
    };
    let fx = fixture(vec![spec("a")], loader, PoolLimits::default());
    let pool = Arc::clone(&fx.pool);
    let loading = tokio::spawn(async move { pool.load("a").await });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let error = fx.pool.unload("a", true).await.unwrap_err();

    assert!(
        matches!(
            error,
            ModelError::Busy {
                state: ModelState::Loading,
                ..
            }
        ),
        "{error}"
    );
    loading.await.unwrap().unwrap();
}

#[tokio::test]
async fn unload_of_an_unknown_model_is_reported() {
    let fx = fixture(
        vec![spec("a")],
        FakeLoader::default(),
        PoolLimits::default(),
    );

    let error = fx.pool.unload("nope", false).await.unwrap_err();

    assert!(matches!(error, ModelError::Unknown(_)));
}

// -- Status -----------------------------------------------------------------

#[tokio::test]
async fn status_reports_lifecycle_facts() {
    let fx = fixture(
        vec![with_ttl("a", 30), pinned("b")],
        FakeLoader::with_sizes(&[("a", 3)]),
        PoolLimits::default(),
    );
    let before = fx.pool.status("a").unwrap();
    assert_eq!(before.state, ModelState::Unloaded);
    assert_eq!(before.last_used_at, None);
    assert_eq!(before.idle_ttl_seconds, Some(30));
    assert_eq!(before.engine, "simple");
    assert!(fx.pool.status("b").unwrap().pinned);

    let lease = fx.pool.acquire("a").await.unwrap();

    let during = fx.pool.status("a").unwrap();
    assert_eq!(during.state, ModelState::Loaded);
    assert_eq!(during.in_flight, 1);
    assert_eq!(during.estimated_bytes, Some(3 * GIB));
    assert_eq!(during.measured_bytes, Some(3 * GIB));
    assert!(during.load_duration_ms.is_some());
    assert!(during.last_used_at.is_some());

    drop(lease);
    assert_eq!(fx.pool.status("a").unwrap().in_flight, 0);
    fx.pool.unload("a", false).await.unwrap();
    let after = fx.pool.status("a").unwrap();
    assert_eq!(after.measured_bytes, None);
    assert_eq!(after.load_duration_ms, None);
}

#[tokio::test]
async fn statuses_are_sorted_by_name() {
    let fx = fixture(
        vec![spec("zebra"), spec("alpha")],
        FakeLoader::default(),
        PoolLimits::default(),
    );

    let names: Vec<String> = fx.pool.statuses().into_iter().map(|s| s.name).collect();

    assert_eq!(names, ["alpha", "zebra"]);
}

#[tokio::test]
async fn preferred_name_favors_a_loaded_model() {
    let fx = fixture(
        vec![spec("a"), spec("b")],
        FakeLoader::default(),
        PoolLimits::default(),
    );
    assert_eq!(fx.pool.preferred_name(), Some("a"));

    touch(&fx.pool, "b").await;

    assert_eq!(fx.pool.preferred_name(), Some("b"));
}

// -- Policy -----------------------------------------------------------------

fn candidate(name: &str, age_secs: u64, now: Instant) -> Candidate {
    Candidate {
        name: name.to_owned(),
        pinned: false,
        in_flight: 0,
        idle_since: now.checked_sub(Duration::from_secs(age_secs)).unwrap(),
        footprint_bytes: Some(GIB),
    }
}

fn incoming(name: &str, reserve_gib: u64) -> Incoming<'_> {
    Incoming {
        name,
        reserve_bytes: reserve_gib * GIB,
        model_bytes: Some(reserve_gib * GIB),
    }
}

#[test]
fn policy_picks_the_oldest_idle_model() {
    let now = Instant::now();
    let resident = [
        candidate("new", 1, now),
        candidate("old", 50, now),
        candidate("mid", 20, now),
    ];

    let decision = decide(&cap(3), incoming("x", 1), 3 * GIB, &resident);

    assert_eq!(decision, Decision::Evict("old".to_owned()));
}

#[test]
fn policy_breaks_ties_by_name() {
    let now = Instant::now();
    let resident = [candidate("b", 5, now), candidate("a", 5, now)];

    let decision = decide(&cap(2), incoming("x", 1), 0, &resident);

    assert_eq!(decision, Decision::Evict("a".to_owned()));
}

#[test]
fn policy_skips_pinned_and_busy_models() {
    let now = Instant::now();
    let pinned_oldest = Candidate {
        pinned: true,
        ..candidate("pinned", 100, now)
    };
    let busy = Candidate {
        in_flight: 2,
        ..candidate("busy", 90, now)
    };
    let free = candidate("free", 1, now);

    let decision = decide(&cap(3), incoming("x", 1), 0, &[pinned_oldest, busy, free]);

    assert_eq!(decision, Decision::Evict("free".to_owned()));
}

#[test]
fn policy_fits_without_limits() {
    let now = Instant::now();

    let decision = decide(
        &PoolLimits::default(),
        incoming("x", 100),
        u64::MAX,
        &[candidate("a", 1, now)],
    );

    assert_eq!(decision, Decision::Fits);
}

#[test]
fn policy_refusal_classifies_blockers() {
    let now = Instant::now();
    let pinned_model = Candidate {
        pinned: true,
        ..candidate("p", 1, now)
    };
    let busy = Candidate {
        in_flight: 3,
        ..candidate("b", 1, now)
    };

    let Decision::Refuse(refusal) = decide(&cap(2), incoming("x", 1), 0, &[pinned_model, busy])
    else {
        panic!("expected a refusal");
    };

    assert_eq!(
        refusal
            .resident
            .iter()
            .map(|r| r.blocker)
            .collect::<Vec<_>>(),
        [Blocker::InUse(3), Blocker::Pinned]
    );
    assert!(refusal.over_count);
    assert!(!refusal.over_memory);
}
