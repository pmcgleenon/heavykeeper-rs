#![warn(missing_docs)]
//! HeavyKeeper is for finding Top-K elephant flows with high precision and low memory footprint
//!
//! This implementation is based on the paper [HeavyKeeper: An Accurate
//! Algorithm for Finding Top-k Elephant Flows][paper] by Junzhi Gong, Tong
//! Yang, Haowei Zhang, and Hao Li, Peking University; Steve Uhlig, Queen Mary,
//! University of London; Shigang Chen, University of Florida; Lorna Uden,
//! Staffordshire University; Xiaoming Li, Peking University.
//!
//! [paper]: https://www.usenix.org/system/files/conference/atc18/atc18-gong.pdf
//!
//! # Variants
//!
//! - [`TopK`]: the algorithm from the paper, with `depth` rows of `width`
//!   buckets.
//! - [`BucketedTopK`]: `width` buckets of `depth` cells; each item hashes to
//!   one bucket.
//! - [`CuckooTopK`]: `width` buckets, each with one decaying lobby cell and
//!   `depth` heavy slots; heavy items move between two candidate buckets.
//!
//! # Parameters
//!
//! - `k`: the number of items to track.
//! - `width`: buckets per row ([`TopK`]) or number of buckets. A power of two
//!   avoids a modulo when indexing.
//! - `depth`: rows ([`TopK`]), cells per bucket ([`BucketedTopK`]), or heavy
//!   slots per bucket ([`CuckooTopK`]).
//! - `decay`: the decay base, in `0.0..=1.0`. When an item collides with a
//!   cell held by another item with count `c`, that count is decremented with
//!   probability `decay^c`.
//!
//! The builders validate `width`, `depth`, and `decay` and return an error;
//! the infallible constructors (`new`, `with_seed`, `with_hasher`) trust the
//! caller.
//!
//! # Example
//!
//! ```
//! use heavykeeper::TopK;
//!
//! let mut topk: TopK<String> = TopK::new(2, 1024, 4, 0.9);
//!
//! topk.add("apple", 10);
//! topk.add("banana", 5);
//! topk.add("cherry", 1);
//!
//! let top = topk.list();
//! assert_eq!(top.len(), 2);
//! assert_eq!(top[0].item, "apple");
//! assert_eq!(top[0].count, 10);
//! assert_eq!(top[1].item, "banana");
//! ```
//!
//! # Merging and serialization
//!
//! Sketches can only be merged if they have the same parameters and hasher,
//! for example the same seed (`new` uses a fixed seed). A builder given
//! neither a seed nor a hasher picks a random hasher.
//!
//! Sketches of `Vec<u8>` items can be written with `to_bytes`. The hasher is
//! not stored, so `from_bytes` needs the original seed, and a mismatch returns
//! [`DeserializeError::HasherMismatch`]. aHash output depends on the CPU
//! architecture and aHash version, so bytes may not load on a different one.

mod heavykeeper;
pub use heavykeeper::{
    Builder, BuilderError, HeavyKeeperError, TopK, TopKDeserializeError, TopKNode,
};

mod bucketed;
pub use bucketed::{
    BucketedBuilder, BucketedBuilderError, BucketedDeserializeError, BucketedMergeError,
    BucketedNode, BucketedTopK,
};

mod cuckoo;
pub use cuckoo::{
    CuckooBuilder, CuckooBuilderError, CuckooDeserializeError, CuckooMergeError, CuckooNode,
    CuckooTopK, DEFAULT_MAX_CUCKOO_KICKS,
};

mod hash_composition;
mod priority_queue;

mod serialization;
pub use serialization::DeserializeError;
