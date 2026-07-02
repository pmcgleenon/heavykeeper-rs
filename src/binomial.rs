use rand::Rng;

/// Sample from Binomial(n, p) in O(1) expected time.
///
/// Uses direct simulation for n ≤ 10, normal approximation via Box-Muller
/// for np(1-p) > 5.0, and Poisson approximation (Knuth) otherwise.
/// The p > 0.5 case is handled via complement symmetry.
pub(crate) fn sample_binomial(n: u64, p: f64, rng: &mut impl Rng) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    if p > 0.5 {
        return n - sample_binomial(n, 1.0 - p, rng);
    }

    if n <= 10 {
        let mut k = 0;
        for _ in 0..n {
            if rng.random_bool(p) {
                k += 1;
            }
        }
        return k;
    }

    let np = n as f64 * p;
    let npq = np * (1.0 - p);

    if npq > 5.0 {
        let u1: f64 = rng.random_range(0.0..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let k = (np + npq.sqrt() * z).round();
        return k.max(0.0).min(n as f64) as u64;
    }

    let lambda = (-np).exp();
    let mut k = 0u64;
    let mut prod = 1.0;
    loop {
        k += 1;
        prod *= rng.random::<f64>();
        if prod <= lambda {
            break;
        }
    }
    (k - 1).min(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn old(incr: u64, cur: u64, p: f64, rng: &mut impl Rng) -> bool {
        let mut remaining = incr;
        let mut c = cur;
        while remaining > 0 && c > 0 {
            if rng.random_bool(p) {
                c -= 1;
                if c == 0 {
                    return true;
                }
            }
            remaining -= 1;
        }
        false
    }

    #[test]
    fn test_binomial_scales_better() {
        use std::time::Instant;

        fn old_ms(incr: u64, cur: u64, p: f64, trials: u32) -> f64 {
            let t0 = Instant::now();
            for _ in 0..trials {
                old(incr, cur, p, &mut SmallRng::seed_from_u64(42));
            }
            t0.elapsed().as_secs_f64() * 1000.0 / trials as f64
        }

        fn new_ms(incr: u64, _cur: u64, p: f64, trials: u32) -> f64 {
            let t0 = Instant::now();
            for _ in 0..trials {
                sample_binomial(incr, p, &mut SmallRng::seed_from_u64(42));
            }
            t0.elapsed().as_secs_f64() * 1000.0 / trials as f64
        }

        let sizes = [100u64, 1_000, 10_000, 100_000];
        let trials = [500u32, 100, 20, 5];

        println!(
            "\n{:<10} {:>8} {:>14} {:>14} {:>10}",
            "n", "trials", "old (ms)", "new (ms)", "speedup"
        );
        println!("{}", "-".repeat(61));

        for (&n, &t) in sizes.iter().zip(trials.iter()) {
            let cur = n / 2;
            let old_ms = old_ms(n, cur, 0.5, t);
            let new_ms = new_ms(n, cur, 0.5, t);
            let speedup = if new_ms > 0.0 {
                old_ms / new_ms
            } else {
                f64::INFINITY
            };
            println!(
                "{:<10} {:>8} {:>14.4} {:>14.4} {:>9.1}×",
                n, t, old_ms, new_ms, speedup
            );
        }
    }

    #[test]
    fn test_sample_binomial_edge_cases() {
        let mut rng = SmallRng::seed_from_u64(42);
        // n = 0 → always 0
        assert_eq!(sample_binomial(0, 0.5, &mut rng), 0);
        // p = 0.0 → always 0
        assert_eq!(sample_binomial(100, 0.0, &mut rng), 0);
        // p = 1.0 → always n
        assert_eq!(sample_binomial(100, 1.0, &mut rng), 100);
        // Large n with p = 1.0 (triggers early return before all paths)
        assert_eq!(sample_binomial(1_000_000, 1.0, &mut rng), 1_000_000);
    }

    #[test]
    fn test_sample_binomial_small_n() {
        let mut rng = SmallRng::seed_from_u64(42);
        for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for _ in 0..100 {
                let k = sample_binomial(5, p, &mut rng);
                assert!(k <= 5, "sample_binomial(5, {p}) = {k} > 5");
            }
        }
        // With p = 1.0 every trial must yield exactly n.
        for _ in 0..20 {
            assert_eq!(sample_binomial(5, 1.0, &mut rng), 5);
        }
    }

    #[test]
    fn test_sample_binomial_normal_approx() {
        // n=1000, p=0.5 → npq = 250 > 5 → Box-Muller normal path
        let mut rng = SmallRng::seed_from_u64(42);
        let trials = 10_000;
        let mut sum = 0u64;
        for _ in 0..trials {
            sum += sample_binomial(1000, 0.5, &mut rng);
        }
        let mean = sum as f64 / trials as f64;
        // Expected mean = 500.  SEM ≈ sqrt(250/10000) ≈ 0.16, so ±10 is very generous.
        assert!(
            (mean - 500.0).abs() < 10.0,
            "sample_binomial(1000, 0.5) mean = {mean}, expected ~500"
        );
    }

    #[test]
    fn test_sample_binomial_poisson_approx() {
        // n=200, p=0.01 → npq = 1.98 ≤ 5 → Poisson (Knuth) path
        let mut rng = SmallRng::seed_from_u64(42);
        let trials = 10_000;
        let mut sum = 0u64;
        for _ in 0..trials {
            sum += sample_binomial(200, 0.01, &mut rng);
        }
        let mean = sum as f64 / trials as f64;
        // Expected mean ≈ 2.0.  SEM ≈ sqrt(1.98/10000) ≈ 0.014, so ±1 is generous.
        assert!(
            (mean - 2.0).abs() < 1.0,
            "sample_binomial(200, 0.01) mean = {mean}, expected ~2.0"
        );
    }

    #[test]
    fn test_sample_binomial_complement_symmetry() {
        let mut rng = SmallRng::seed_from_u64(42);
        // For p > 0.5, the function uses n - sample_binomial(n, 1-p).
        // Verify results stay in [0, n] for a range of (n, p).
        for n in [1, 3, 10, 100, 1000] {
            for p in [0.0, 0.3, 0.5, 0.7, 1.0] {
                for _ in 0..20 {
                    let k = sample_binomial(n, p, &mut rng);
                    assert!(
                        k <= n,
                        "sample_binomial({n}, {p}) = {k} > {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_decays_zero_branch() {
        // When p = 0.0 (decay threshold = 0), sample_binomial always returns 0.
        // Verify the calling code (simulated here) does nothing.
        let mut rng = SmallRng::seed_from_u64(42);
        assert_eq!(sample_binomial(u64::MAX, 0.0, &mut rng), 0);
        assert_eq!(sample_binomial(1_000_000, 0.0, &mut rng), 0);
        assert_eq!(sample_binomial(0, 0.0, &mut rng), 0);
    }
}
