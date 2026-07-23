//! Trait definitions for configurable fingerprint and counter widths.
//!
//! By default, `TopK`, `BucketedTopK`, and `CuckooTopK` use `u64` for both
//! fingerprint and count storage (16 bytes per cell). For memory-constrained
//! deployments, narrower types can be used:
//!
//! - `u32` fingerprint + `u32` count = 8 bytes/cell (2× savings)
//! - `u16` fingerprint + `u16` count = 4 bytes/cell (4× savings)
//!
//! # Example
//! ```
//! use heavykeeper::TopK;
//!
//! // Default: u64 fingerprint + u64 count (16 bytes/cell)
//! let topk: TopK<String> = TopK::new(10, 8192, 2, 0.95);
//!
//! // Compact: u32 fingerprint + u32 count (8 bytes/cell)
//! let topk: TopK<String, u32, u32> = TopK::new(10, 8192, 2, 0.95);
//!
//! // Minimal: u16 fingerprint + u16 count (4 bytes/cell)
//! let topk: TopK<String, u16, u16> = TopK::new(10, 8192, 2, 0.95);
//! ```

use std::fmt::Debug;

/// Trait for types usable as a bucket fingerprint.
///
/// A fingerprint is a compact hash used to identify which item occupies a
/// bucket. Wider fingerprints reduce false positive collisions but cost more
/// memory per cell.
///
/// - `u64`: 1-in-2^64 collision rate (overkill for most workloads)
/// - `u32`: 1-in-4-billion collision rate per cell
/// - `u16`: 1-in-65536 collision rate per cell (sufficient for top-K)
pub trait Fingerprint: Copy + Clone + Default + Debug + PartialEq + Eq + Send + Sync + 'static {
    /// Truncate a full u64 hash value into this fingerprint width.
    fn from_hash(h: u64) -> Self;
}

/// Trait for types usable as a bucket counter.
///
/// A counter tracks how many times the fingerprinted item has been seen.
/// Wider counters support higher maximum counts but cost more memory per cell.
///
/// - `u64`: counts up to 2^64 (overkill)
/// - `u32`: counts up to ~4 billion (sufficient for virtually all workloads)
/// - `u16`: counts up to 65535 (sufficient for many top-K workloads with decay)
pub trait Counter: Copy + Clone + Default + Debug + PartialOrd + Ord + PartialEq + Eq + Send + Sync + 'static {
    /// The zero value.
    const ZERO: Self;

    /// The maximum representable value (used for saturation in `from_u64`).
    const MAX: Self;

    /// Add `rhs` with saturation at MAX.
    fn saturating_add(self, rhs: Self) -> Self;

    /// Subtract `rhs` with saturation at ZERO.
    fn saturating_sub(self, rhs: Self) -> Self;

    /// Convert from u64, saturating at Self::MAX.
    fn from_u64(v: u64) -> Self;

    /// Widen to u64 for decay probability math and priority queue interaction.
    fn as_u64(self) -> u64;

    /// Return self - 1, saturating at zero.
    #[inline]
    fn dec(self) -> Self {
        self.saturating_sub(Self::from_u64(1))
    }

    /// Check if this is zero.
    #[inline]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

// --- Fingerprint implementations ---

impl Fingerprint for u64 {
    #[inline]
    fn from_hash(h: u64) -> Self {
        h
    }
}

impl Fingerprint for u32 {
    #[inline]
    fn from_hash(h: u64) -> Self {
        // Use upper 32 bits (lower bits used for bucket indexing)
        (h >> 32) as u32
    }
}

impl Fingerprint for u16 {
    #[inline]
    fn from_hash(h: u64) -> Self {
        // Use bits 48..64 (highest 16 bits, least correlated with bucket index)
        (h >> 48) as u16
    }
}

// --- Counter implementations ---

impl Counter for u64 {
    const ZERO: Self = 0;
    const MAX: Self = u64::MAX;

    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        u64::saturating_add(self, rhs)
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        u64::saturating_sub(self, rhs)
    }

    #[inline]
    fn from_u64(v: u64) -> Self {
        v
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self
    }
}

impl Counter for u32 {
    const ZERO: Self = 0;
    const MAX: Self = u32::MAX;

    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        u32::saturating_add(self, rhs)
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        u32::saturating_sub(self, rhs)
    }

    #[inline]
    fn from_u64(v: u64) -> Self {
        if v > u32::MAX as u64 {
            u32::MAX
        } else {
            v as u32
        }
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl Counter for u16 {
    const ZERO: Self = 0;
    const MAX: Self = u16::MAX;

    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        u16::saturating_add(self, rhs)
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        u16::saturating_sub(self, rhs)
    }

    #[inline]
    fn from_u64(v: u64) -> Self {
        if v > u16::MAX as u64 {
            u16::MAX
        } else {
            v as u16
        }
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_u64_roundtrip() {
        let h: u64 = 0xDEAD_BEEF_CAFE_BABE;
        assert_eq!(<u64 as Fingerprint>::from_hash(h), h);
    }

    #[test]
    fn test_fingerprint_u32_takes_upper_bits() {
        let h: u64 = 0xDEAD_BEEF_CAFE_BABE;
        assert_eq!(<u32 as Fingerprint>::from_hash(h), 0xDEAD_BEEF);
    }

    #[test]
    fn test_fingerprint_u16_takes_highest_bits() {
        let h: u64 = 0xDEAD_BEEF_CAFE_BABE;
        assert_eq!(<u16 as Fingerprint>::from_hash(h), 0xDEAD);
    }

    #[test]
    fn test_counter_u16_saturates() {
        let c = u16::MAX;
        assert_eq!(c.saturating_add(1), u16::MAX);
        assert_eq!(<u16 as Counter>::from_u64(100_000), u16::MAX);
    }

    #[test]
    fn test_counter_u32_saturates() {
        let c = u32::MAX;
        assert_eq!(c.saturating_add(1), u32::MAX);
        assert_eq!(<u32 as Counter>::from_u64(5_000_000_000), u32::MAX);
    }

    #[test]
    fn test_counter_is_zero() {
        assert!(<u64 as Counter>::ZERO.is_zero());
        assert!(<u32 as Counter>::ZERO.is_zero());
        assert!(<u16 as Counter>::ZERO.is_zero());
        assert!(!1u64.is_zero());
    }
}
