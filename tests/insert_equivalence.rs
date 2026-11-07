//! Pins every `add_with_evicted` result and the final `list()` of each sketch over seeded
//! skewed and uniform streams, so changes to the insert fast path cannot alter behavior.

use heavykeeper::{BucketedTopK, CuckooTopK, TopK};

struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn result(&mut self, (evicted, inserted): (Option<u64>, bool)) {
        self.u64(evicted.map_or(u64::MAX, |e| e));
        self.u64(u64::from(inserted));
    }
}

/// Skewed stream: squaring a uniform value concentrates mass on small keys.
fn stream(seed: u64, len: usize, universe: u64, skewed: bool) -> Vec<u64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let u = if skewed { u * u * u } else { u };
            (u * universe as f64) as u64
        })
        .collect()
}

const CASES: &[(usize, usize, bool)] = &[
    (4096, 4, true),
    (4096, 4, false),
    (1000, 3, true),
    (1000, 3, false),
];

fn digests() -> Vec<u64> {
    let mut out = vec![];
    for &(width, depth, skewed) in CASES {
        let items = stream(
            0x9e37_79b9_7f4a_7c15 ^ width as u64,
            200_000,
            50_000,
            skewed,
        );

        let mut d = Digest::new();
        let mut topk = TopK::<u64>::with_seed(100, width, depth, 0.9, 7);
        for (n, item) in items.iter().enumerate() {
            d.result(topk.add_with_evicted(item, 1 + (n % 3 == 0) as u64));
        }
        for node in topk.list() {
            d.u64(node.item);
            d.u64(node.count);
        }
        out.push(d.0);

        let mut d = Digest::new();
        let mut bucketed = BucketedTopK::<u64>::with_seed(100, width, depth, 0.9, 7);
        for (n, item) in items.iter().enumerate() {
            d.result(bucketed.add_with_evicted(item, 1 + (n % 3 == 0) as u64));
        }
        for node in bucketed.list() {
            d.u64(node.item);
            d.u64(node.count);
        }
        out.push(d.0);

        let mut d = Digest::new();
        let mut cuckoo = CuckooTopK::<u64>::with_seed(100, width, depth, 0.9, 7);
        for (n, item) in items.iter().enumerate() {
            d.result(cuckoo.add_with_evicted(item, 1 + (n % 3 == 0) as u64));
        }
        for node in cuckoo.list() {
            d.u64(node.item);
            d.u64(node.count);
        }
        out.push(d.0);
    }
    out
}

const EXPECTED: &[u64] = &[
    0x74dae6970ab7b261,
    0x79dafd6e632cd4df,
    0x79dafd6e632cd4df,
    0xffece9194c25c80c,
    0x39d296fea06a9aba,
    0x5e1137b16970040e,
    0x3b9b6c1f049e2c1d,
    0x08c9930813bf40ee,
    0xaf53ec720b076f71,
    0xbe6d2d412a0427e8,
    0xcd13d081c9aaa02d,
    0xe69e8d82047de2b7,
];

#[test]
fn insert_results_are_stable() {
    assert_eq!(digests(), EXPECTED);
}
