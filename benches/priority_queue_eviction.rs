#[allow(dead_code, unused_imports)]
#[path = "../src/priority_queue.rs"]
mod priority_queue;

use ahash::RandomState;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use priority_queue::TopKQueue;
use std::hint::black_box;

const CAPACITY: usize = 64;
const INSERTS: usize = 100_000;

fn make_keys(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut key = vec![0u8; len];
            let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for byte in &mut key {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                *byte = x as u8;
            }
            key[..8.min(len)].copy_from_slice(&i.to_le_bytes()[..8.min(len)]);
            key
        })
        .collect()
}

fn bench_replacing_upserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue_eviction");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.throughput(criterion::Throughput::Elements(INSERTS as u64));

    for key_len in [16usize, 64, 256, 1024] {
        let keys = make_keys(INSERTS, key_len);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("vec_len_{key_len}")),
            &keys,
            |b, keys| {
                b.iter_batched(
                    || TopKQueue::with_capacity_and_hasher(CAPACITY, RandomState::new()),
                    |mut queue| {
                        let mut evictions = 0usize;
                        for (i, key) in keys.iter().enumerate() {
                            evictions += queue
                                .upsert(black_box(key.clone()), (i + 1) as u64)
                                .is_some() as usize;
                        }
                        black_box((queue.len(), evictions));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_replacing_upserts);
criterion_main!(benches);
