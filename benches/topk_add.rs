use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand_distr::Distribution;
use zipf::ZipfDistribution;

use heavykeeper::TopK;

fn benchmark_topk_add(c: &mut Criterion, num_adds: usize) {
    let mut rng = rand::thread_rng();
    let zipf = ZipfDistribution::new(100_000, 1.03).unwrap();
    let mut topk = TopK::new(10, 1024, 2, 0.95);

    let mut data = vec![];
    for _ in 0..num_adds {
        let key = zipf.sample(&mut rng);
        data.push(key);
    }

    let mut group = c.benchmark_group(format!("TopK_Add_{}", num_adds));
    group.sample_size(60); // Reduce the sample count
    group.warm_up_time(std::time::Duration::from_secs(3)); // Increase the warm-up time
    group.measurement_time(std::time::Duration::from_secs(10)); // Increase the measurement time

    group.bench_function("Add", |b| {
        b.iter(|| {
            for &key in data.iter() {
                topk.add(black_box(key));
            }
        });
    });
    group.finish();
}

criterion_group!(benches,
    benchmark_topk_add_1,
    benchmark_topk_add_10,
    benchmark_topk_add_100,
    benchmark_topk_add_1_000,
    benchmark_topk_add_10_000,
    benchmark_topk_add_100_000,
    benchmark_topk_add_1_000_000
);
criterion_main!(benches);

fn benchmark_topk_add_1(c: &mut Criterion) {
    benchmark_topk_add(c, 1);
}

fn benchmark_topk_add_10(c: &mut Criterion) {
    benchmark_topk_add(c, 10);
}

fn benchmark_topk_add_100(c: &mut Criterion) {
    benchmark_topk_add(c, 100);
}

fn benchmark_topk_add_1_000(c: &mut Criterion) {
    benchmark_topk_add(c, 1_000);
}

fn benchmark_topk_add_10_000(c: &mut Criterion) {
    benchmark_topk_add(c, 10_000);
}

fn benchmark_topk_add_100_000(c: &mut Criterion) {
    benchmark_topk_add(c, 100_000);
}

fn benchmark_topk_add_1_000_000(c: &mut Criterion) {
    benchmark_topk_add(c, 1_000_000);
}

