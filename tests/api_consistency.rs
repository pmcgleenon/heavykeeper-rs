//! Regression tests for API inconsistencies between the three sketches.

use heavykeeper::{
    BucketedBuilder, BucketedTopK, Builder, BuilderError, CuckooBuilder, CuckooTopK,
    HeavyKeeperError, TopK, DEFAULT_MAX_CUCKOO_KICKS,
};

/// Builder and error types returned by public APIs must be nameable by callers.
#[test]
fn builder_and_error_types_are_nameable() {
    let _: Builder<u64> = TopK::<u64>::builder();
    let _: BucketedBuilder<u64> = BucketedTopK::<u64>::builder();
    let _: CuckooBuilder<u64> = CuckooTopK::<u64>::builder();
    let _: usize = DEFAULT_MAX_CUCKOO_KICKS;

    let mut a = TopK::<u64>::with_seed(10, 64, 2, 0.9, 1);
    let b = TopK::<u64>::with_seed(10, 128, 2, 0.9, 1);
    let err: HeavyKeeperError = a.merge(&b).unwrap_err();
    assert!(matches!(err, HeavyKeeperError::IncompatibleWidth { .. }));
}

#[test]
fn topk_builder_rejects_invalid_parameters() {
    let base = || {
        TopK::<u64>::builder()
            .k(10)
            .width(64)
            .depth(2)
            .decay(0.9)
            .seed(1)
    };
    assert!(matches!(
        base().width(0).build(),
        Err(BuilderError::InvalidWidth { width: 0 })
    ));
    assert!(matches!(
        base().depth(0).build(),
        Err(BuilderError::InvalidDepth { depth: 0 })
    ));
    for decay in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                base().decay(decay).build(),
                Err(BuilderError::InvalidDecay { .. })
            ),
            "decay {decay}"
        );
    }
    assert!(base().build().is_ok());
}

#[test]
fn topk_counts_saturate_instead_of_overflowing() {
    let mut topk = TopK::<u64>::with_seed(10, 64, 2, 0.9, 1);
    topk.add(&7, u64::MAX);
    topk.add(&7, 1);
    assert_eq!(topk.count(&7), u64::MAX);

    let mut merged = TopK::<u64>::with_seed(10, 64, 2, 0.9, 1);
    merged.add(&7, u64::MAX);
    let other = merged.clone();
    merged.merge(&other).unwrap();
    assert_eq!(merged.count(&7), u64::MAX);
    assert_eq!(merged.list()[0].count, u64::MAX);
}

/// `with_seed` must seed the decay RNG like the seeded builder does, as `TopK` and
/// `CuckooTopK` already do.
#[test]
fn bucketed_with_seed_seeds_the_decay_rng() {
    let stream: Vec<u64> = (0..20_000u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9) % 5_000)
        .collect();
    let run = |mut sketch: BucketedTopK<u64>| {
        for item in &stream {
            sketch.add(item, 1);
        }
        // Per-item counts read the cells, which record every decay decision.
        (0..5_000u64)
            .map(|item| sketch.count(&item))
            .collect::<Vec<_>>()
    };
    let from_seed = run(BucketedTopK::with_seed(50, 64, 2, 0.9, 42));
    let from_builder = run(BucketedTopK::builder()
        .k(50)
        .width(64)
        .depth(2)
        .decay(0.9)
        .seed(42)
        .build()
        .unwrap());
    assert_eq!(from_seed, from_builder);
}

/// `TopK::merge` must credit an item tracked by only one side with the other
/// side's bucket count, as `BucketedTopK` and `CuckooTopK` do.
#[test]
fn topk_merge_counts_items_tracked_by_one_side() {
    let sketch = || TopK::<String>::with_seed(2, 1024, 4, 0.9, 1);
    let (mut a, mut b) = (sketch(), sketch());
    // `a` tracks x and y, with w only in its buckets; `b` tracks z and w,
    // with y only in its buckets. The true top 2 is w (95) and z (60).
    for (item, count) in [("x", 55), ("y", 50), ("w", 45)] {
        a.add(item, count);
    }
    for (item, count) in [("z", 60), ("w", 50), ("y", 5)] {
        b.add(item, count);
    }
    let top = |s: &TopK<String>| {
        s.list()
            .into_iter()
            .map(|n| (n.item, n.count))
            .collect::<Vec<_>>()
    };
    let expected = vec![("w".to_string(), 95), ("z".to_string(), 60)];

    let mut ab = a.clone();
    ab.merge(&b).unwrap();
    assert_eq!(top(&ab), expected);

    let mut ba = b.clone();
    ba.merge(&a).unwrap();
    assert_eq!(top(&ba), expected);
}
