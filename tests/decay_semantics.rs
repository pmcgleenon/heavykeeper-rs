//! Regression tests for batched-add decay semantics.
//!
//! HeavyKeeper's `add(item, n)` is defined as the sequential per-unit
//! algorithm: it must behave as if the item arrived `n` times, one at a
//! time. Each unit of increment is one decay trial against a colliding
//! bucket; the decay probability is re-evaluated as the incumbent's count
//! drops, and when the count reaches zero the challenger takes the bucket
//! with the *unconsumed remainder* of the increment.

use heavykeeper::{BucketedTopK, CuckooTopK, TopK};

/// With decay = 1.0 every trial decays, so the whole process is
/// deterministic and batched vs. single-unit adds must agree exactly.
///
/// item1 holds the bucket with count 1000. Adding item2 with weight 3000
/// spends 1000 units grinding item1 to zero, takes the bucket with the
/// remainder, and ends at count 2001 — identical to calling
/// add(item2, 1) 3000 times.
#[test]
fn batched_add_matches_single_unit_adds() {
    let item1 = b"item1".to_vec();
    let item2 = b"item2".to_vec();

    // width=1, depth=1: both items share the single bucket.
    let mut batched: TopK<Vec<u8>> = TopK::new(1, 1, 1, 1.0);
    batched.add(&item1, 1000);
    batched.add(&item2, 3000);

    let mut singles: TopK<Vec<u8>> = TopK::new(1, 1, 1, 1.0);
    singles.add(&item1, 1000);
    for _ in 0..3000 {
        singles.add(&item2, 1);
    }

    let b = batched.list();
    let s = singles.list();
    assert_eq!(b[0].item, item2);
    assert_eq!(s[0].item, item2);
    assert_eq!(
        b[0].count, s[0].count,
        "add(item, 3000) must agree with 3000 x add(item, 1): batched={} singles={}",
        b[0].count, s[0].count
    );
    assert_eq!(
        b[0].count, 2001,
        "takeover count must be the unconsumed remainder of the increment"
    );
}

/// A challenger whose weight is several times the expected eviction cost
/// must take over the bucket.
///
/// item1 holds the bucket with count 120 (decay 0.9). The sequential
/// process needs ~3.1M units on average to grind that to zero
/// (sum of 0.9^-c for c in 1..=120), because the decay probability rises
/// as the count drops. A 15M-unit add therefore evicts with overwhelming
/// probability (failure odds < 1e-9, and the RNG is seeded, so the run is
/// deterministic).
///
/// An implementation that freezes the decay probability at the initial
/// count (p = 0.9^120 ~= 3.2e-6) expects only ~48 decays from 15M trials
/// but needs 120 — a >10 sigma shortfall — so it essentially never evicts.
#[test]
fn heavy_challenger_evicts_incumbent() {
    let item1 = b"item1".to_vec();
    let item2 = b"item2".to_vec();

    let mut topk: TopK<Vec<u8>> = TopK::new(1, 1, 1, 0.9);
    topk.add(&item1, 120);
    topk.add(&item2, 15_000_000);

    let nodes = topk.list();
    assert_eq!(
        nodes[0].item, item2,
        "a 15M-weight challenger must evict a 120-count incumbent"
    );
}

/// The motivating bug for the batched decay path: a colliding add with a
/// huge increment must not iterate once per unit. The incumbent's count
/// is high enough (decay^1e6 ~= 0) that the challenger cannot win the
/// bucket, so the entire add is failed decay trials — the worst case for
/// a per-unit loop, which would run u64::MAX iterations here.
#[test]
fn huge_increment_add_does_not_hang_topk() {
    let mut topk: TopK<Vec<u8>> = TopK::new(1, 1, 1, 0.9);
    topk.add(&b"alpha".to_vec(), 1_000_000);
    topk.add(&b"beta".to_vec(), u64::MAX);
    assert_eq!(topk.list()[0].item, b"alpha".to_vec());
}

#[test]
fn huge_increment_add_does_not_hang_bucketed() {
    let mut topk: BucketedTopK<Vec<u8>> = BucketedTopK::new(10, 1, 1, 0.9);
    topk.add(&b"alpha".to_vec(), 1_000_000);
    topk.add(&b"beta".to_vec(), u64::MAX);
    assert!(topk.list().len() <= 10);
}

#[test]
fn huge_increment_add_does_not_hang_cuckoo() {
    let mut topk: CuckooTopK<Vec<u8>> = CuckooTopK::new(10, 1, 1, 0.9);
    topk.add(&b"hot".to_vec(), 1_000_000_000); // fills the heavy slot
    topk.add(&b"alpha".to_vec(), 1_000_000); // stays in the lobby
    topk.add(&b"beta".to_vec(), u64::MAX); // decays the occupied lobby
    assert!(topk.list().len() <= 10);
}
