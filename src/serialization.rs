//! Shared byte-serialization error, constants, and readers for all variants.

use thiserror::Error;

/// Error returned by every variant's `from_bytes` (aliased per variant).
#[derive(Error, Debug)]
pub enum DeserializeError {
    /// The input ended before a field could be read in full.
    #[error(
        "Byte stream too short while reading {field}: need {needed} byte(s), have {actual}"
    )]
    UnexpectedEof {
        /// Name of the field being read.
        field: &'static str,
        /// Number of bytes the field requires.
        needed: usize,
        /// Number of bytes that were left in the input.
        actual: usize,
    },

    /// The input does not start with the `HVYK` magic bytes.
    #[error("Not a heavykeeper sketch: bad magic bytes {actual:02x?} (expected {expected:02x?})")]
    BadMagic {
        /// The expected magic bytes (`b"HVYK"`).
        expected: [u8; 4],
        /// The first four bytes of the input.
        actual: [u8; 4],
    },

    /// The input was written by a different sketch type.
    #[error("Payload is a different sketch variant: got tag {actual} (expected {expected})")]
    WrongVariant {
        /// Variant tag of the sketch type being restored.
        expected: u8,
        /// Variant tag found in the input.
        actual: u8,
    },

    /// The supplied seed or hasher does not hash like the one that wrote the
    /// input.
    #[error("Hasher mismatch: the supplied hasher produces probe {actual} but the payload holds {expected} (wrong seed, a different hasher, or a different ahash version/architecture)")]
    HasherMismatch {
        /// Probe hash stored in the input.
        expected: u64,
        /// Probe hash produced by the supplied seed or hasher.
        actual: u64,
    },

    /// The input uses a serialization format version this build cannot read.
    #[error("Unsupported serialization version {version} (this build expects {expected})")]
    UnsupportedVersion {
        /// Format version found in the input.
        version: u8,
        /// Format version this build reads and writes.
        expected: u8,
    },

    /// A field holds an out-of-range value.
    #[error("Invalid {field} value: {detail}")]
    InvalidField {
        /// Name of the invalid field.
        field: &'static str,
        /// Description of the problem.
        detail: String,
    },

    /// A stored length exceeds the limit set by the sketch parameters.
    #[error("Length mismatch for {field}: payload holds {actual} but expected {expected}")]
    LengthMismatch {
        /// Name of the field.
        field: &'static str,
        /// Length stored in the input.
        actual: usize,
        /// Maximum length allowed by the sketch parameters.
        expected: usize,
    },

    /// The input has extra bytes after the end of the sketch.
    #[error("{count} unexpected trailing byte(s) after the sketch payload")]
    TrailingBytes {
        /// Number of extra bytes.
        count: usize,
    },
}

/// Magic tag at the start of every serialized sketch (`b"HVYK"`).
pub(crate) const MAGIC: [u8; 4] = *b"HVYK";
/// On-disk format version.
///
/// Bump whenever the byte layout changes, OR when the interpretation of any
/// field changes even though no bytes move — including the RNG algorithm
/// behind `rng_state` (see `fastrand_stream_is_pinned`) and the hash family
/// behind the fingerprints and `hasher_probe`. A payload's `rng_state` cannot
/// be translated across RNG algorithms; a future loader migrating an old
/// version should reuse the stored bytes as a fresh seed (replication pairs
/// must full-resync across such an upgrade).
pub(crate) const VERSION: u8 = 1;
/// Probe hashed at serialize time to detect a wrong seed on load.
///
/// `ahash` output is not stable across CPU architectures or `ahash` versions,
/// so a payload only loads on the same architecture and `ahash` version that
/// wrote it; otherwise the probe mismatches and load fails with `HasherMismatch`.
pub(crate) const SERIALIZE_HASHER_PROBE: &[u8] = b"heavykeeper-serialize-hasher-probe";
/// Bytes per serialized cell: `(fingerprint: u64, count: u64)`.
pub(crate) const CELL_SIZE: usize = 16;
/// Bytes in a serialized `fastrand::Rng` state (its 64-bit seed, little-endian).
pub(crate) const RNG_STATE_SIZE: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Cell {
    pub(crate) fingerprint: u64,
    pub(crate) count: u64,
}

/// Parse a `CELL_SIZE`-aligned slice into a boxed cell array.
pub(crate) fn parse_cells(slice: &[u8]) -> Box<[Cell]> {
    slice
        .chunks_exact(CELL_SIZE)
        .map(|chunk| Cell {
            fingerprint: u64::from_le_bytes(chunk[0..8].try_into().expect("8 bytes")),
            count: u64::from_le_bytes(chunk[8..16].try_into().expect("8 bytes")),
        })
        .collect()
}

/// A forward-only cursor over a serialized payload. Every read is
/// bounds-checked and advances the cursor, so `from_bytes` never touches raw
/// offsets and a truncated stream fails with a precise `UnexpectedEof`.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read `n` bytes, advancing the cursor.
    pub(crate) fn take(
        &mut self,
        n: usize,
        field: &'static str,
    ) -> Result<&'a [u8], DeserializeError> {
        let available = self.bytes.len().saturating_sub(self.pos);
        if available < n {
            return Err(DeserializeError::UnexpectedEof {
                field,
                needed: n,
                actual: available,
            });
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read a fixed-size byte array.
    pub(crate) fn take_array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], DeserializeError> {
        Ok(self.take(N, field)?.try_into().expect("slice is N bytes"))
    }

    /// Read a single byte.
    pub(crate) fn take_u8(&mut self, field: &'static str) -> Result<u8, DeserializeError> {
        Ok(self.take(1, field)?[0])
    }

    /// Read a little-endian `u64`.
    pub(crate) fn take_u64(&mut self, field: &'static str) -> Result<u64, DeserializeError> {
        Ok(u64::from_le_bytes(self.take_array::<8>(field)?))
    }

    /// Read a little-endian `u64` and narrow it to `usize`, erroring on overflow.
    pub(crate) fn take_usize(&mut self, field: &'static str) -> Result<usize, DeserializeError> {
        let value = self.take_u64(field)?;
        usize::try_from(value).map_err(|_| DeserializeError::InvalidField {
            field,
            detail: format!("value {value} exceeds usize range on this platform"),
        })
    }

    /// Verify the fixed header shared by every variant: magic, `variant` tag,
    /// version, and the hasher probe. `probe` is the caller's hash of
    /// [`SERIALIZE_HASHER_PROBE`] with the hasher it intends to restore with;
    /// a mismatch against the stored probe rejects the payload before any
    /// params are parsed. Taking the probe (not a seed) keeps this module
    /// hasher-agnostic — any `BuildHasher` can produce one via `hash_one`.
    pub(crate) fn read_header(&mut self, variant: u8, probe: u64) -> Result<(), DeserializeError> {
        let magic = self.take_array::<4>("magic")?;
        if magic != MAGIC {
            return Err(DeserializeError::BadMagic {
                expected: MAGIC,
                actual: magic,
            });
        }
        let got_variant = self.take_u8("variant")?;
        if got_variant != variant {
            return Err(DeserializeError::WrongVariant {
                expected: variant,
                actual: got_variant,
            });
        }
        let version = self.take_u8("version")?;
        if version != VERSION {
            return Err(DeserializeError::UnsupportedVersion {
                version,
                expected: VERSION,
            });
        }
        let stored_probe = self.take_u64("hasher_probe")?;
        if probe != stored_probe {
            return Err(DeserializeError::HasherMismatch {
                expected: stored_probe,
                actual: probe,
            });
        }
        Ok(())
    }

    /// Read and validate the params shared by every variant
    pub(crate) fn read_params(&mut self) -> Result<(usize, usize, f64, usize), DeserializeError> {
        let width = self.take_usize("width")?;
        let depth = self.take_usize("depth")?;
        let decay = f64::from_bits(self.take_u64("decay")?);
        let top_items = self.take_usize("top_items")?;

        if width < 1 {
            return Err(DeserializeError::InvalidField {
                field: "width",
                detail: format!("must be >= 1, got {width}"),
            });
        }
        if depth < 1 {
            return Err(DeserializeError::InvalidField {
                field: "depth",
                detail: format!("must be >= 1, got {depth}"),
            });
        }
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(DeserializeError::InvalidField {
                field: "decay",
                detail: format!("must be a finite value in 0.0..=1.0, got {decay}"),
            });
        }

        Ok((width, depth, decay, top_items))
    }

    /// Reject any bytes left after the payload.
    pub(crate) fn finish(&self) -> Result<(), DeserializeError> {
        if self.pos != self.bytes.len() {
            return Err(DeserializeError::TrailingBytes {
                count: self.bytes.len() - self.pos,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::RandomState;

    const SEED: u64 = 42;
    const VARIANT: u8 = 0;

    /// Probe for `seed`, matching what `from_bytes` computes.
    fn probe_for(seed: u64) -> u64 {
        RandomState::with_seeds(seed, seed, seed, seed).hash_one(SERIALIZE_HASHER_PROBE)
    }

    /// Build a valid header (magic, variant, version, probe) for `SEED`.
    fn header(variant: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(variant);
        out.push(VERSION);
        out.extend_from_slice(&probe_for(SEED).to_le_bytes());
        out
    }

    #[test]
    fn read_header_rejects_bad_magic() {
        let mut bytes = header(VARIANT);
        bytes[0] ^= 0xff;
        let mut r = ByteReader::new(&bytes);
        assert!(matches!(
            r.read_header(VARIANT, probe_for(SEED)),
            Err(DeserializeError::BadMagic { .. })
        ));
    }

    #[test]
    fn read_header_rejects_wrong_variant() {
        let bytes = header(VARIANT + 1);
        let mut r = ByteReader::new(&bytes);
        assert!(matches!(
            r.read_header(VARIANT, probe_for(SEED)),
            Err(DeserializeError::WrongVariant { .. })
        ));
    }

    #[test]
    fn read_header_rejects_unsupported_version() {
        let mut bytes = header(VARIANT);
        bytes[5] = VERSION + 1;
        let mut r = ByteReader::new(&bytes);
        assert!(matches!(
            r.read_header(VARIANT, probe_for(SEED)),
            Err(DeserializeError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn read_header_rejects_wrong_seed() {
        let bytes = header(VARIANT);
        let mut r = ByteReader::new(&bytes);
        assert!(matches!(
            r.read_header(VARIANT, probe_for(SEED + 1)),
            Err(DeserializeError::HasherMismatch { .. })
        ));
    }

    #[test]
    fn read_header_rejects_truncated() {
        let bytes = header(VARIANT);
        let mut r = ByteReader::new(&bytes[..bytes.len() - 1]);
        assert!(matches!(
            r.read_header(VARIANT, probe_for(SEED)),
            Err(DeserializeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn read_params_validates_scalars() {
        // width, depth, decay, top_items
        let mut ok = Vec::new();
        ok.extend_from_slice(&8u64.to_le_bytes());
        ok.extend_from_slice(&4u64.to_le_bytes());
        ok.extend_from_slice(&0.9f64.to_bits().to_le_bytes());
        ok.extend_from_slice(&10u64.to_le_bytes());
        let mut r = ByteReader::new(&ok);
        assert_eq!(r.read_params().unwrap(), (8, 4, 0.9, 10));

        // width = 0 is rejected.
        let mut bad = ok.clone();
        bad[0..8].copy_from_slice(&0u64.to_le_bytes());
        let mut r = ByteReader::new(&bad);
        assert!(matches!(
            r.read_params(),
            Err(DeserializeError::InvalidField { field: "width", .. })
        ));

        // depth = 0 is rejected.
        let mut bad = ok.clone();
        bad[8..16].copy_from_slice(&0u64.to_le_bytes());
        let mut r = ByteReader::new(&bad);
        assert!(matches!(
            r.read_params(),
            Err(DeserializeError::InvalidField { field: "depth", .. })
        ));

        // out-of-range decay is rejected.
        let mut bad = ok.clone();
        bad[16..24].copy_from_slice(&2.0f64.to_bits().to_le_bytes());
        let mut r = ByteReader::new(&bad);
        assert!(matches!(
            r.read_params(),
            Err(DeserializeError::InvalidField { field: "decay", .. })
        ));
    }

    #[test]
    fn finish_rejects_trailing_bytes() {
        let bytes = [0u8; 2];
        let mut r = ByteReader::new(&bytes);
        r.take(1, "x").unwrap();
        assert!(matches!(
            r.finish(),
            Err(DeserializeError::TrailingBytes { count: 1 })
        ));
    }

    /// Golden-stream test: pins `fastrand`'s output for a fixed seed.
    ///
    /// A payload's `rng_state` only keeps a restored replica in decay-lockstep
    /// with its primary if both draw the same stream from the same state. Every
    /// other test compares a sketch against itself, so a fastrand release that
    /// changed its generator (as 1.x -> 2.0 did) would pass the whole suite
    /// while silently breaking mixed-version replication.
    ///
    /// If this fails: the RNG stream changed. Bump [`VERSION`] (see its doc for
    /// the migration rule) — do not just update the constants.
    #[test]
    fn fastrand_stream_is_pinned() {
        let mut rng = fastrand::Rng::with_seed(42);
        assert_eq!(
            [rng.u64(..), rng.u64(..), rng.u64(..)],
            [
                0xca71d87c76983989,
                0x7e5ba61552085fc6,
                0xcdf101e3bab88b9f,
            ],
            "fastrand's output stream changed for a fixed seed"
        );
        // `get_seed` must expose the raw state: with_seed(get_seed()) is how
        // `from_bytes` resumes the stream exactly.
        let mut a = fastrand::Rng::with_seed(7);
        let _ = a.u64(..);
        let mut b = fastrand::Rng::with_seed(a.get_seed());
        assert_eq!(a.u64(..), b.u64(..), "state round-trip must resume the stream");
    }

    /// Golden-bytes test: pins the serialized layout of an empty `TopK`.
    ///
    /// The round-trip tests can't catch a format change that alters write and
    /// read symmetrically; this hardcodes the bytes so any layout drift fails
    /// loudly. The `hasher_probe` field (bytes 6..14) is checked against a
    /// live hash rather than a constant because ahash output is
    /// architecture-dependent; everything else is fixed.
    ///
    /// If this fails: the on-disk format changed. Bump [`VERSION`] and update
    /// the layout docs on every variant's `to_bytes` — do not just update the
    /// constants.
    #[test]
    fn topk_layout_is_pinned() {
        let bytes = crate::TopK::<Vec<u8>>::with_seed(3, 2, 1, 0.9, SEED).to_bytes();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"HVYK"); // magic
        expected.push(0); // variant: TopK
        expected.push(1); // version
        expected.extend_from_slice(&probe_for(SEED).to_le_bytes()); // hasher_probe
        expected.extend_from_slice(&2u64.to_le_bytes()); // width
        expected.extend_from_slice(&1u64.to_le_bytes()); // depth
        expected.extend_from_slice(&0x3FECCCCCCCCCCCCDu64.to_le_bytes()); // decay = 0.9f64 bits
        expected.extend_from_slice(&3u64.to_le_bytes()); // top_items
        expected.extend_from_slice(&[0u8; 2 * 16]); // 2x1 empty cells
        expected.extend_from_slice(&0u64.to_le_bytes()); // pq_len
        expected.extend_from_slice(&42u64.to_le_bytes()); // rng_state = unadvanced seed

        assert_eq!(bytes, expected, "TopK serialized layout changed");
    }

    /// Golden-bytes test for `CuckooTopK` (the variant valkey-bloom persists):
    /// same rules as `topk_layout_is_pinned`.
    #[test]
    fn cuckoo_layout_is_pinned() {
        let bytes = crate::CuckooTopK::<Vec<u8>>::with_seed(3, 2, 1, 0.9, SEED).to_bytes();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"HVYK"); // magic
        expected.push(2); // variant: CuckooTopK
        expected.push(1); // version
        expected.extend_from_slice(&probe_for(SEED).to_le_bytes()); // hasher_probe
        expected.extend_from_slice(&2u64.to_le_bytes()); // width
        expected.extend_from_slice(&1u64.to_le_bytes()); // depth
        expected.extend_from_slice(&0x3FECCCCCCCCCCCCDu64.to_le_bytes()); // decay = 0.9f64 bits
        expected.extend_from_slice(&3u64.to_le_bytes()); // top_items
        expected.extend_from_slice(&8u64.to_le_bytes()); // max_kicks (default)
        expected.extend_from_slice(&[0u8; 2 * 16]); // 2 empty lobby cells
        expected.extend_from_slice(&[0u8; 2 * 16]); // 2x1 empty heavy cells
        expected.extend_from_slice(&0u64.to_le_bytes()); // pq_len
        expected.extend_from_slice(&42u64.to_le_bytes()); // rng_state = unadvanced seed

        assert_eq!(bytes, expected, "CuckooTopK serialized layout changed");
    }
}
