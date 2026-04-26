//! Side-by-side accuracy: canonical `TopK` vs `BucketedTopK`.
//! Metrics: Top-K Hit Ratio and Average Relative Error (HeavyKeeper paper).
//! Run with `cargo test --test accuracy_compare --release -- --nocapture`.

use std::collections::{HashMap, HashSet};

use heavykeeper::{BucketedTopK, TopK};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Zipf};

const ZIPF_N: f64 = 1_000_000.0;
const STREAM_LEN: usize = 5_000_000;
const K: usize = 100;
// 1024 cells (256*4) for 1M distinct keys — tight enough to force eviction.
const WIDTH: usize = 256;
const DEPTH: usize = 4;
const DECAY: f64 = 0.9;
const SEED: u64 = 0xACC04ACC;

fn gen_zipf(s: f64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Zipf::new(ZIPF_N, s).expect("valid zipf");
    (0..STREAM_LEN).map(|_| dist.sample(&mut rng) as u64).collect()
}

fn ground_truth(stream: &[u64]) -> HashMap<u64, u64> {
    let mut m = HashMap::with_capacity(stream.len());
    for k in stream {
        *m.entry(*k).or_insert(0u64) += 1;
    }
    m
}

fn true_top_k(truth: &HashMap<u64, u64>) -> Vec<(u64, u64)> {
    let mut v: Vec<_> = truth.iter().map(|(k, c)| (*k, *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(K);
    v
}

struct Metrics {
    hit_ratio: f64,
    are: f64,
}

fn metrics_topk(stream: &[u64], truth: &HashMap<u64, u64>, true_set: &HashSet<u64>) -> Metrics {
    let mut sketch: TopK<u64> = TopK::with_seed(K, WIDTH, DEPTH, DECAY, 12345);
    for k in stream {
        sketch.add(k, 1);
    }
    score(&sketch.list(), truth, true_set, |n| (n.item, n.count))
}

fn metrics_bucketed(stream: &[u64], truth: &HashMap<u64, u64>, true_set: &HashSet<u64>) -> Metrics {
    let mut sketch: BucketedTopK<u64> = BucketedTopK::with_seed(K, WIDTH, DEPTH, DECAY, 12345);
    for k in stream {
        sketch.add(k, 1);
    }
    score(&sketch.list(), truth, true_set, |n| (n.item, n.count))
}

fn score<N>(
    reported: &[N],
    truth: &HashMap<u64, u64>,
    true_set: &HashSet<u64>,
    extract: impl Fn(&N) -> (u64, u64),
) -> Metrics {
    let pairs: Vec<(u64, u64)> = reported.iter().map(&extract).collect();
    let hits = pairs.iter().filter(|(item, _)| true_set.contains(item)).count();
    let hit_ratio = hits as f64 / K as f64;

    // ARE skips false positives — true count of 0 has no meaningful ratio.
    let mut sum = 0.0;
    let mut n = 0;
    for (item, est) in &pairs {
        if let Some(&true_c) = truth.get(item) {
            if true_c > 0 {
                sum += (*est as f64 - true_c as f64).abs() / true_c as f64;
                n += 1;
            }
        }
    }
    let are = if n == 0 { 0.0 } else { sum / n as f64 };
    Metrics { hit_ratio, are }
}

fn run_case(s: f64) -> (Metrics, Metrics) {
    let stream = gen_zipf(s, SEED);
    let truth = ground_truth(&stream);
    let true_top = true_top_k(&truth);
    let true_set: HashSet<u64> = true_top.iter().map(|(k, _)| *k).collect();

    let m_topk = metrics_topk(&stream, &truth, &true_set);
    let m_bkt = metrics_bucketed(&stream, &truth, &true_set);

    println!(
        "  Zipf s={s:<5}  TopK         hit_ratio={:.4}  ARE={:.6}",
        m_topk.hit_ratio, m_topk.are
    );
    println!(
        "  Zipf s={s:<5}  BucketedTopK hit_ratio={:.4}  ARE={:.6}",
        m_bkt.hit_ratio, m_bkt.are
    );

    (m_topk, m_bkt)
}

#[test]
fn compare_accuracy_zipf() {
    println!();
    println!(
        "Stream: {STREAM_LEN} items, ZIPF_N={ZIPF_N}, k={K}, width={WIDTH}, depth={DEPTH}, decay={DECAY}"
    );
    println!();

    let (t1, b1) = run_case(2.0);
    assert!(t1.hit_ratio >= 0.80, "TopK s=2.0 hit_ratio {} < 0.80", t1.hit_ratio);
    assert!(b1.hit_ratio >= 0.80, "BucketedTopK s=2.0 hit_ratio {} < 0.80", b1.hit_ratio);

    let (t2, b2) = run_case(1.2);
    assert!(t2.hit_ratio >= 0.50, "TopK s=1.2 hit_ratio {} < 0.50", t2.hit_ratio);
    assert!(b2.hit_ratio >= 0.50, "BucketedTopK s=1.2 hit_ratio {} < 0.50", b2.hit_ratio);

    let (t3, b3) = run_case(1.05);
    assert!(t3.hit_ratio >= 0.20, "TopK s=1.05 hit_ratio {} < 0.20", t3.hit_ratio);
    assert!(b3.hit_ratio >= 0.20, "BucketedTopK s=1.05 hit_ratio {} < 0.20", b3.hit_ratio);

    for (label, m) in [
        ("TopK 2.0", &t1), ("BucketedTopK 2.0", &b1),
        ("TopK 1.2", &t2), ("BucketedTopK 1.2", &b2),
    ] {
        assert!(m.are < 1.0, "{label} ARE {} >= 1.0", m.are);
    }
}
