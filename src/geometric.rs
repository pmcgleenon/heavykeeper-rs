use rand::Rng;

/// Sample the trial index of the first success in a sequence of Bernoulli
/// trials that each succeed when `rng.next_u64() < threshold` (success
/// probability `threshold / 2^64`), without simulating the trials.
///
/// Uses inversion: `G = ceil(ln(U) / ln(1 - p))` with `U` uniform in
/// `(0, 1]`, so `G` is geometric on `{1, 2, ...}`. This lets the decay
/// loop skip straight to the next successful decay in O(1) instead of
/// spending one RNG draw per unit of increment.
///
/// Returns `u64::MAX` when the first success lies beyond any practical
/// budget (the inversion overflows for tiny `p`); callers treat a result
/// larger than their remaining budget as "no decay in this add".
/// `threshold == 0` (never succeeds) must be short-circuited by the caller.
pub(crate) fn sample_geometric(threshold: u64, rng: &mut impl Rng) -> u64 {
    let p = threshold as f64 / (u64::MAX as f64);
    if p >= 1.0 {
        return 1;
    }
    // 1 - [0, 1) = (0, 1]: avoids ln(0).
    let u = 1.0 - rng.random::<f64>();
    // ln(1 - p) via ln_1p keeps precision when p is tiny.
    let g = (u.ln() / (-p).ln_1p()).ceil();
    if g < 1.0 {
        1
    } else if g >= u64::MAX as f64 {
        u64::MAX
    } else {
        g as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn certain_success_takes_one_trial() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(sample_geometric(u64::MAX, &mut rng), 1);
        }
    }

    #[test]
    fn tiny_threshold_saturates() {
        // p ~= 5.4e-20: the first success is astronomically far away and
        // must not wrap or panic.
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            assert!(sample_geometric(1, &mut rng) > 1_000_000_000_000);
        }
    }

    #[test]
    fn mean_matches_one_over_p() {
        // p = 0.25 -> E[G] = 4. With 100k samples the sample mean has
        // SEM ~= sqrt(12)/sqrt(100k) ~= 0.011, so +/-0.15 is generous.
        let threshold = u64::MAX / 4;
        let mut rng = SmallRng::seed_from_u64(42);
        let trials = 100_000u64;
        let sum: u64 = (0..trials)
            .map(|_| sample_geometric(threshold, &mut rng))
            .sum();
        let mean = sum as f64 / trials as f64;
        assert!(
            (mean - 4.0).abs() < 0.15,
            "sample_geometric(p=0.25) mean = {mean}, expected ~4.0"
        );
    }

    #[test]
    fn matches_per_trial_simulation() {
        // The whole point: G must be distributed like the index of the
        // first success in per-trial simulation. Compare survival at a
        // few points against the exact CDF for p = 0.1.
        let threshold = u64::MAX / 10;
        let mut rng = SmallRng::seed_from_u64(7);
        let trials = 100_000u64;
        let mut le_10 = 0u64;
        for _ in 0..trials {
            if sample_geometric(threshold, &mut rng) <= 10 {
                le_10 += 1;
            }
        }
        // P(G <= 10) = 1 - 0.9^10 ~= 0.6513; SEM ~= 0.0015.
        let frac = le_10 as f64 / trials as f64;
        assert!(
            (frac - 0.6513).abs() < 0.01,
            "P(G <= 10) = {frac}, expected ~0.6513"
        );
    }
}
