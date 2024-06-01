use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand_distr::Distribution;
use zipf::ZipfDistribution;

use heavykeeper::TopK;

fn benchmark_topk_add(c: &mut Criterion) {

    let mut rng = rand::thread_rng();
    let zipf = ZipfDistribution::new(100_000, 1.03).unwrap();
    let mut topk = TopK::new(10, 1024, 5, 0.95);

    let mut data = vec![];
    for _ in 0..1_000_000 {
        let key = zipf.sample(&mut rng);
        data.push(key);
    }

    let mut group = c.benchmark_group("TopK_Add");
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

criterion_group!(benches, benchmark_topk_add);
criterion_main!(benches);

