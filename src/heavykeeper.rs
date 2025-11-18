use ahash::RandomState;
use std::borrow::Borrow;
use std::clone::Clone;
use std::fmt::Debug;
use std::hash::Hash;
use rand::{SeedableRng, RngCore};
use rand::rngs::SmallRng;
use thiserror::Error;
use crate::priority_queue::TopKQueue;
use crate::hash_composition::HashComposer;

const DECAY_LOOKUP_SIZE: usize = 1024;

#[derive(Default, Clone, Debug)]
struct Bucket {
    fingerprint: u64,
    count: u64,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node<T> {
    pub item: T,
    pub count: u64,
}

impl<T: Ord> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.count.cmp(&self.count) // Reverse ordering for min-heap
    }
}

impl<T: Ord> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Error, Debug)]
pub enum HeavyKeeperError {
    #[error("Incompatible width: self ({self_width}) != other ({other_width})")]
    IncompatibleWidth {
        self_width: usize,
        other_width: usize,
    },
    
    #[error("Incompatible depth: self ({self_depth}) != other ({other_depth})")]
    IncompatibleDepth {
        self_depth: usize,
        other_depth: usize,
    },
    
    #[error("Incompatible decay: self ({self_decay}) != other ({other_decay})")]
    IncompatibleDecay {
        self_decay: f64,
        other_decay: f64,
    },
    
    #[error("Incompatible top_items: self ({self_items}) != other ({other_items})")]
    IncompatibleTopItems {
        self_items: usize,
        other_items: usize,
    },
}

#[derive(Error, Debug)]
pub enum BuilderError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },
}

pub struct TopK<T: Ord + Clone + Hash + Debug> {
    top_items: usize,
    width: usize,
    depth: usize,
    decay: f64,
    decay_thresholds: Vec<u64>,
    buckets: Vec<Vec<Bucket>>,
    priority_queue: TopKQueue<T>,
    hasher: RandomState,
    random: Box<dyn RngCore + Send + Sync>,
}

pub struct Builder<T> {
    k: Option<usize>,
    width: Option<usize>,
    depth: Option<usize>,
    decay: Option<f64>,
    seed: Option<u64>,
    hasher: Option<RandomState>,
    rng: Option<Box<dyn RngCore + Send + Sync>>,
    _phantom: std::marker::PhantomData<T>,
}

fn precompute_decay_thresholds(decay: f64, num_entries: usize) -> Vec<u64> {
    let mut thresholds = Vec::with_capacity(num_entries);
    for count in 0..num_entries {
        let decay_factor = decay.powf(count as f64);
        // Use full u64 range so decay_factor = 1.0 gives probability ~= 1.0 (not 0.5)
        let threshold = (decay_factor * u64::MAX as f64) as u64;
        thresholds.push(threshold);
    }
    thresholds
}

impl<T: Ord + Clone + Hash + Debug> TopK<T> {
    pub fn builder() -> Builder<T> {
        Builder::new()
    }

    pub fn new(k: usize, width: usize, depth: usize, decay: f64) -> Self {
        // Use a consistent seed for default initialization
        let seed = 12345; // Arbitrary but fixed seed
        Self::with_seed(k, width, depth, decay, seed)
    }

    // New constructor that takes a seed
    pub fn with_seed(k: usize, width: usize, depth: usize, decay: f64, seed: u64) -> Self {
        let hasher = RandomState::with_seeds(seed, seed, seed, seed);
        Self::with_hasher(k, width, depth, decay, hasher)
    }

    pub fn with_hasher(k: usize, width: usize, depth: usize, decay: f64, hasher: RandomState) -> Self {
        Self::with_components(k, width, depth, decay, hasher, Box::new(SmallRng::seed_from_u64(0)))
    }

    fn with_components(k: usize, width: usize, depth: usize, decay: f64, hasher: RandomState, rng: Box<dyn RngCore + Send + Sync>) -> Self {
        // Pre-allocate with capacity to avoid resizing
        let mut buckets = Vec::with_capacity(depth);
        for _ in 0..depth {
            buckets.push(vec![Bucket { fingerprint: 0, count: 0 }; width]);
        }

        Self {
            top_items: k,
            width,
            depth,
            decay,
            decay_thresholds: precompute_decay_thresholds(decay, DECAY_LOOKUP_SIZE),
            buckets,
            priority_queue: TopKQueue::with_capacity_and_hasher(k, hasher.clone()),
            hasher,
            random: rng,
        }
    }

    pub fn query<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        if self.priority_queue.get(item).is_some() {
            return true;
        }

        let mut composer = HashComposer::new(&self.hasher, item);
        let mut min_count = u64::MAX;

        for i in 0..self.depth {
            let bucket_idx = composer.next_bucket(self.width as u64, i);
            let bucket = &self.buckets[i][bucket_idx];

            if bucket.fingerprint == composer.fingerprint() {
                min_count = min_count.min(bucket.count);
            }
        }

        min_count != u64::MAX
    }

    pub fn count<Q>(&self, item: &Q) -> u64
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        if let Some(count) = self.priority_queue.get(item) {
            return count;
        }

        let mut composer = HashComposer::new(&self.hasher, item);
        let mut min_count = u64::MAX;

        for i in 0..self.depth {
            let bucket_idx = composer.next_bucket(self.width as u64, i);
            let bucket = &self.buckets[i][bucket_idx];

            if bucket.fingerprint == composer.fingerprint() {
                min_count = min_count.min(bucket.count);
            }
        }

        if min_count == u64::MAX {
            0
        } else {
            min_count
        }
    }

    #[cfg(test)]
    pub fn bucket_count<Q>(&self, item: &Q) -> u64
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        let mut composer = HashComposer::new(&self.hasher, item);
        let mut min_count = u64::MAX;

        for i in 0..self.depth {
            let bucket_idx = composer.next_bucket(self.width as u64, i);
            let bucket = &self.buckets[i][bucket_idx];

            if bucket.fingerprint == composer.fingerprint() {
                min_count = min_count.min(bucket.count);
            }
        }

        if min_count == u64::MAX {
            0
        } else {
            min_count
        }
    }

    pub fn add<Q>(&mut self, item: &Q, increment: u64)
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        let mut composer = HashComposer::new(&self.hasher, item);
        let mut max_count: u64 = 0;

        for i in 0..self.depth {
            let bucket_idx = composer.next_bucket(self.width as u64, i);
            let bucket = &mut self.buckets[i][bucket_idx];

            let matches = bucket.fingerprint == composer.fingerprint();
            let empty = bucket.count == 0u64;
            
            if matches || empty {
                bucket.fingerprint = composer.fingerprint();
                bucket.count += increment;
                max_count = std::cmp::max(max_count, bucket.count);
            } else {
                let mut remaining_incr = increment;
                while remaining_incr > 0 {
                    let count_idx = bucket.count as usize;
                    let decay_threshold = if count_idx < self.decay_thresholds.len() {
                        self.decay_thresholds[count_idx]
                    } else {

                        let lookup_size = self.decay_thresholds.len();
                        // Keep extrapolation normalized to the same [0,1] scale as precomputed thresholds.
                        let base_threshold = self.decay_thresholds[lookup_size - 1] as f64 / u64::MAX as f64;
                        let divisor = lookup_size - 1;
                        let quotient = count_idx / divisor;
                        let remainder = count_idx % divisor;
                        let remainder_threshold = self.decay_thresholds[remainder] as f64 / u64::MAX as f64;
                        let decay = base_threshold.powi(quotient as i32) * remainder_threshold;
                        // Re-scale back into u64 range consistent with precomputed thresholds.
                        (decay * u64::MAX as f64) as u64
                    };
                    let rand = self.random.next_u64();
                    if rand < decay_threshold {
                        bucket.count = bucket.count.saturating_sub(1);

                        if bucket.count == 0 {
                            bucket.fingerprint = composer.fingerprint();
                            bucket.count = remaining_incr;
                            max_count = std::cmp::max(max_count, bucket.count);
                            break;
                        }
                    }

                    remaining_incr -= 1;
                }
            }
        }

        // First check if queue is full - this is a cheap O(1) operation
        if self.priority_queue.is_full() {
            // Only check min_count if queue is full
            if max_count < self.priority_queue.min_count() {
                return;
            }
        }

        // Clone the item here since we need to store it in the priority queue
        self.priority_queue.upsert(item.to_owned(), max_count);
    }

    pub fn list(&self) -> Vec<Node<T>> {
        let mut nodes = self.priority_queue.iter().map(|(item, count)| Node {
            item: item.clone(),
            count,
        }).collect::<Vec<_>>();
        nodes.sort();
        nodes
    }

    pub fn debug(&self) {
        println!("width: {}", self.width);
        println!("depth: {}", self.depth);
        println!("decay: {}", self.decay);
        println!("decay thresholds: {:?}", self.decay_thresholds);
        let mut buckets: Vec<(&Bucket, usize, usize)> = self
            .buckets
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(j, bucket)| (bucket, i, j))
            })
            .filter(|(bucket, _, _)| bucket.count != 0)
            .collect();
        buckets.sort_by(|a, b| b.0.count.cmp(&a.0.count));
        for (bucket, i, j) in buckets {
            println!("Bucket at row {}, column {}: {:?}", i, j, bucket);
        }
        println!("priority_queue: ");
        let mut nodes = self.priority_queue.iter().map(|(item, count)| Node {
            item: item.clone(),
            count,
        }).collect::<Vec<_>>();

        nodes.sort();
        for node in nodes {
            println!("Node - Item: {:?}, Count: {}", node.item, node.count);
        }
    }

    // Merge another HeavyKeeper into this one
    pub fn merge(&mut self, other: &Self) -> Result<(), HeavyKeeperError> {
        // Verify compatible parameters
        if self.width != other.width {
            return Err(HeavyKeeperError::IncompatibleWidth {
                self_width: self.width,
                other_width: other.width,
            });
        }
        
        if self.depth != other.depth {
            return Err(HeavyKeeperError::IncompatibleDepth {
                self_depth: self.depth,
                other_depth: other.depth,
            });
        }
        
        if self.decay != other.decay {
            return Err(HeavyKeeperError::IncompatibleDecay {
                self_decay: self.decay,
                other_decay: other.decay,
            });
        }

        if self.top_items != other.top_items {
            return Err(HeavyKeeperError::IncompatibleTopItems {
                self_items: self.top_items,
                other_items: other.top_items,
            });
        }

        // Merge bucket counts
        for (self_row, other_row) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            for (self_bucket, other_bucket) in self_row.iter_mut().zip(other_row.iter()) {
                if self_bucket.fingerprint == other_bucket.fingerprint {
                    // Same item, add counts
                    self_bucket.count += other_bucket.count;
                } else if self_bucket.count == 0 {
                    // Empty bucket in self, copy from other
                    *self_bucket = other_bucket.clone();
                }
                // If different items and self bucket not empty, keep existing item
            }
        }

        // Merge priority queues
        for (item, count) in other.priority_queue.iter() {
            let self_count = self.priority_queue.get(item).unwrap_or(0);
            self.priority_queue.upsert(item.clone(), self_count + count);
        }

        Ok(())
    }
}

impl<T: Ord + Clone + Hash + Debug> Default for Builder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone + Hash + Debug> Builder<T> {
    pub fn new() -> Self {
        Self {
            k: None,
            width: None,
            depth: None,
            decay: None,
            seed: None,
            hasher: None,
            rng: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn decay(mut self, decay: f64) -> Self {
        self.decay = Some(decay);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn hasher(mut self, hasher: RandomState) -> Self {
        self.hasher = Some(hasher);
        self
    }

    pub fn rng<R: RngCore + Send + Sync + 'static>(mut self, rng: R) -> Self {
        self.rng = Some(Box::new(rng));
        self
    }

    pub fn build(self) -> Result<TopK<T>, BuilderError> {
        let k = self.k.ok_or_else(|| BuilderError::MissingField { field: "k".to_string() })?;
        let width = self.width.ok_or_else(|| BuilderError::MissingField { field: "width".to_string() })?;
        let depth = self.depth.ok_or_else(|| BuilderError::MissingField { field: "depth".to_string() })?;
        let decay = self.decay.ok_or_else(|| BuilderError::MissingField { field: "decay".to_string() })?;

        let hasher = self.hasher.unwrap_or_else(|| {
            if let Some(seed) = self.seed {
                RandomState::with_seeds(seed, seed, seed, seed)
            } else {
                RandomState::new()
            }
        });

        let rng = self.rng.unwrap_or_else(|| {
            if let Some(seed) = self.seed {
                Box::new(SmallRng::seed_from_u64(seed))
            } else {
                Box::new(SmallRng::seed_from_u64(0))
            }
        });

        Ok(TopK::with_components(k, width, depth, decay, hasher, rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::automock;

    #[automock]
    trait RngCoreTrait {
        fn next_u64(&mut self) -> u64;
    }

    impl RngCore for MockRngCoreTrait {
        fn next_u32(&mut self) -> u32 {
            RngCoreTrait::next_u64(self) as u32
        }
        
        fn next_u64(&mut self) -> u64 {
            RngCoreTrait::next_u64(self)
        }
        
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                let value = RngCoreTrait::next_u64(self);
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = (value >> (i * 8)) as u8;
                }
            }
        }
    }

    /// Tests basic initialization of TopK with default parameters
    #[test]
    fn test_new() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);
        assert_eq!(topk.width, 100);
        assert_eq!(topk.depth, 5);
        assert_eq!(topk.decay, 0.9);
        assert_eq!(topk.buckets.len(), 5);
        assert_eq!(topk.buckets[0].len(), 100);
        assert_eq!(topk.priority_queue.len(), 0);
    }

    /// Tests query functionality for both present and absent items
    #[test]
    fn test_query() {
        let mut topk: TopK<Vec<u8>> = TopK::new(10, 100, 5, 0.9);
        let present = b"hello".to_vec();
        let absent = b"world".to_vec();

        // Add the present item
        topk.add(&present, 1);

        // Verify query behavior
        assert!(topk.query(&present), "Present item should be found");
        assert!(!topk.query(&absent), "Absent item should not be found");
    }

    /// Tests count functionality for items with varying frequencies
    #[test]
    fn test_count() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item1 = b"lashin".to_vec();
        let item2 = b"ballynamoney".to_vec();
        let item3 = "पुष्पं अस्ति।".as_bytes().to_vec();

        // Add first item multiple times
        topk.add(&item1, 8);
        assert_eq!(topk.count(&item1), 8, "Count should match number of additions");

        // Verify count for non-existent item
        assert_eq!(topk.count(&item3), 0, "Non-existent item should have count 0");

        // Add second item many times
        topk.add(&item2, 1337);
        assert_eq!(topk.count(&item2), 1337, "Count should match number of additions");
    }

    /// Tests support for non-ASCII characters and emoji
    #[test]
    fn test_non_ascii_and_emoji() {
        let mut topk: TopK<Vec<u8>> = TopK::new(5, 100, 4, 0.9);
        
        // Test with Hindi text
        let p = "पुष्पं अस्ति।".as_bytes().to_vec();
        // Test with emoji
        let emoji = "🚀🌟".as_bytes().to_vec();
        // Test with mixed content
        let mixed = "Hello पुष्पं 🚀".as_bytes().to_vec();

        // Add items
        topk.add(&p, 1);
        topk.add(&emoji, 1);
        topk.add(&mixed, 1);

        // Verify presence
        assert!(topk.query(&p), "text should be found");
        assert!(topk.query(&emoji), "Emoji should be found");
        assert!(topk.query(&mixed), "Mixed content should be found");

        // Verify counts
        assert_eq!(topk.count(&p), 1, "text count should be 1");
        assert_eq!(topk.count(&emoji), 1, "Emoji count should be 1");
        assert_eq!(topk.count(&mixed), 1, "Mixed content count should be 1");

        // Add more occurrences
        topk.add(&p, 4);
        assert_eq!(topk.count(&p), 5, "text count should be 5");

        // Verify display conversion
        let items = topk.list();
        for node in items {
            let text = String::from_utf8_lossy(&node.item);
            println!("Item: {}, Count: {}", text, node.count);
        }
    }

    /// Tests adding a single item and verifying its presence
    #[test]
    fn test_add_single_item() {
        let k = 1;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item = b"hello".to_vec();
        topk.add(&item, 1);

        let nodes = topk.list();
        assert_eq!(nodes.len(), 1, "Should have exactly one item");
        assert_eq!(nodes[0].count, 1, "Count should be 1");
        assert_eq!(nodes[0].item, item, "Item should match");
    }

    /// Tests adding a an item and overwriting it with another
    #[test]
    fn test_add_overwrite() {
        let k = 1;
        let width = 1;
        let depth = 1;
        let decay = 1.0;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // override the decay thresholds so we always decay
        // this removes the probabilistic aspect of decay
        topk.decay_thresholds.iter_mut().for_each(|v| *v = u64::MAX);

        let item1 = b"item1".to_vec();
        topk.add(&item1, 1000);

        let nodes = topk.list();
        assert_eq!(nodes.len(), 1, "Should have exactly one item");
        assert_eq!(nodes[0].count, 1000, "Invalid count");
        assert_eq!(nodes[0].item, item1, "Item should match");

        let item2 = b"item2".to_vec();
        topk.add(&item2, 3000);

        let nodes = topk.list();
        assert_eq!(nodes.len(), 1, "Should have exactly one item");
        assert_eq!(nodes[0].count, 2001, "Invalid count");
        assert_eq!(nodes[0].item, item2, "Item should match");
    }

    /// Tests adding duplicate items and verifying their counts
    #[test]
    fn test_add_duplicate_items() {
        let k = 2; // Track 2 most frequent items
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item1 = b"hello".to_vec();
        let item2 = b"world".to_vec();

        // Add items with equal frequency
        topk.add(&item1, 7);
        topk.add(&item2, 7);

        assert_eq!(topk.priority_queue.len(), k, "Should have exactly k items");

        let nodes = topk.priority_queue.iter().map(|(item, count)| Node {
            item: item.clone(),
            count,
        }).collect::<Vec<_>>();

        assert_eq!(nodes.len(), 2, "Should have exactly two items");
        assert_eq!(nodes[0].count, 7, "First item should have count 7");
        assert_eq!(nodes[0].item, item1, "First item should match");
        assert_eq!(nodes[1].count, 7, "Second item should have count 7");
        assert_eq!(nodes[1].item, item2, "Second item should match");
    }

    /// Tests behavior when adding more items than capacity
    #[test]
    fn test_add_more_items_than_capacity() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let items = [
            b"hello".to_vec(),
            b"world".to_vec(),
            b"ballynamoney".to_vec(),
            b"lane".to_vec(),
        ];

        for item in &items {
            topk.add(item, 1);
        }

        let nodes = topk.list();
        assert_eq!(nodes.len(), 2, "Should maintain capacity limit");
        let mut counts = nodes.iter().map(|node| node.count).collect::<Vec<_>>();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 1], "All items should have count 1");
    }

    /// Tests behavior with different decay values
    #[test]
    fn test_add_with_different_decay() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.5; // Lower decay value for faster count reduction

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let items = [
            b"hello".to_vec(),
            b"world".to_vec(),
            b"ballynamoney".to_vec(),
            b"lane".to_vec(),
            b"pear tree".to_vec(),
        ];

        for item in &items {
            topk.add(item, 1);
        }

        let nodes = topk.list();
        assert_eq!(nodes.len(), 2, "Should maintain capacity limit");
        let mut counts = nodes.iter().map(|node| node.count).collect::<Vec<_>>();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 1], "All items should have count 1");
    }

    /// Tests behavior with empty input
    #[test]
    fn test_add_empty_input() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);
        let nodes = topk.list();
        assert_eq!(nodes.len(), 0, "Should have no items");
    }

    /// Tests behavior with varied input frequencies
    #[test]
    fn test_add_varied_input() {
        let k = 10; // Track top-10 items
        let width = 2000; // Increased width
        let depth = 20;   // Increased depth
        let decay = 0.98; // Higher decay for more stability

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Generate items with increasing frequencies
        let mut items_with_frequencies = Vec::new();
        for i in 0..100 {
            let item = format!("item{}", i);
            let frequency = i + 1;
            items_with_frequencies.push((item, frequency));
        }

        // Add items based on their frequencies
        for (item, frequency) in items_with_frequencies.iter() {
            let item_bytes = item.as_bytes().to_vec();
            for _ in 0..*frequency {
                topk.add(&item_bytes, 1);
            }
        }

        // Verify the priority queue has exactly k items
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Priority queue should contain exactly k items"
        );

        // Print actual top-k for debugging
        let top_items = topk.priority_queue.iter().map(|(item, count)| Node {
            item: std::str::from_utf8(item).unwrap().to_string().into_bytes(),
            count,
        }).collect::<Vec<_>>();

        let expected_top_items = (90..100).map(|i| format!("item{}", i).into_bytes()).collect::<Vec<_>>();

        let mut found = 0;
        for expected_item in expected_top_items.iter() {
            if top_items.iter().any(|node| &node.item == expected_item) {
                found += 1;
            } else {
                println!("Warning: Expected item {} not in top-k", std::str::from_utf8(expected_item).unwrap());
            }
        }
        // Allow at most 2 misses due to sketch randomness
        assert!(found >= 8, "At least 8 of the top 10 items should be in top-k");
    }

    /// Tests behavior with a large number of duplicates
    #[test]
    fn test_large_number_of_duplicates() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item = b"test_item".to_vec();
        let num_additions = 1000;

        // Add the same item many times
        topk.add(&item, num_additions);

        assert_eq!(topk.count(&item), num_additions, "Count should match number of additions");
    }

    /// Tests behavior with multiple distinct items
    #[test]
    fn test_multiple_distinct_items() {
        let k = 2; // Track top-2 items
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item1 = b"item1".to_vec();
        let item2 = b"item2".to_vec();
        let num_additions_item1 = 500;
        let num_additions_item2 = 499; // One less than item1

        // Add items with different frequencies
        topk.add(&item1, num_additions_item1);
        topk.add(&item2, num_additions_item2);

        // Verify counts
        assert_eq!(
            topk.count(&item1),
            num_additions_item1,
            "Count should match number of additions for item1"
        );
        assert_eq!(
            topk.count(&item2),
            num_additions_item2,
            "Count should match number of additions for item2"
        );

        // Verify presence in top-k
        assert!(topk.query(&item1), "item1 should be in top-k");
        assert!(topk.query(&item2), "item2 should be in top-k");
    }

    /// Tests insertion into empty buckets
    #[test]
    fn test_insertion_into_empty_buckets() {
        let k = 5;
        let width = 10;
        let depth = 4;
        let decay = 0.5;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let item = b"new_flow".to_vec();
        topk.add(&item, 1);

        // Verify bucket state
        let item_hash = topk.hasher.hash_one(&item);
        assert!(
            topk.buckets.iter().any(|row| row
                .iter()
                .any(|bucket| bucket.fingerprint == item_hash && bucket.count == 1)),
            "Item should be inserted into an empty bucket with count 1"
        );

        // Verify presence in priority queue
        assert!(topk.query(&item), "Item should be in priority queue");
    }

    /// Tests behavior with items of identical frequency
    #[test]
    fn test_add_identical_frequencies() {
        let k = 10;
        let width = 1000;
        let depth = 10;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Add items with identical frequency
        let frequency = 5;
        for i in 0..100 {
            let item = format!("item{}", i);
            let item_bytes = item.as_bytes().to_vec();
            topk.add(&item_bytes, frequency);
        }

        // Verify priority queue size
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Priority queue should contain exactly k items"
        );

        // Verify all items have the same frequency
        for node in topk.list() {
            assert_eq!(
                node.count,
                frequency,
                "All items should have the same frequency"
            );
        }
    }

    /// Tests behavior with a small k value
    #[test]
    fn test_small_k_value() {
        let k = 2;
        let width = 1000;
        let depth = 10;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Add items with increasing frequencies
        for i in 0..3 {
            let item = format!("item{}", i);
            let item_bytes = item.as_bytes().to_vec();
            topk.add(&item_bytes, i+1);
        }

        // Verify priority queue size
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Priority queue should contain exactly k items"
        );

        // Verify top-k items
        let top_items = topk.priority_queue.iter().map(|(item, count)| Node {
            item: std::str::from_utf8(item).unwrap().to_string().into_bytes(),
            count,
        }).collect::<Vec<_>>();

        let expected_top_items = (1..3).map(|i| format!("item{}", i).into_bytes()).collect::<Vec<_>>();

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {} to be in top-k",
                std::str::from_utf8(expected_item).unwrap()
            );
        }
    }

    /// Tests count functionality with sketch
    #[test]
    fn test_count_with_sketch() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        let items = [
            b"item1".to_vec(),
            b"item2".to_vec(),
            b"item3".to_vec(),
            b"item4".to_vec(),
        ];

        // Add items with different frequencies
        topk.add(&items[0], 1);
        topk.add(&items[1], 1);
        topk.add(&items[2], 2);
        topk.add(&items[3], 5);

        // Verify counts
        assert_eq!(topk.count(&items[0]), 1, "Count should be 1");
        assert_eq!(topk.count(&items[1]), 1, "Count should be 1");
        assert_eq!(topk.count(&items[2]), 2, "Count should be 2");
        assert_eq!(topk.count(&items[3]), 5, "Count should be 5");
    }

    /// Tests basic merge functionality
    #[test]
    fn test_merge_basic() {
        let seed = 12345;
        let mut hk1 = TopK::with_seed(3, 100, 5, 0.9, seed);
        let mut hk2 = TopK::with_seed(3, 100, 5, 0.9, seed);

        let items = [
            b"item1".to_vec(),
            b"item2".to_vec(),
            b"item3".to_vec(),
        ];

        // Add items to first instance
        hk1.add(&items[0], 5);
        hk1.add(&items[1], 3);

        // Add items to second instance
        hk2.add(&items[0], 4);
        hk2.add(&items[2], 6);

        // Merge and verify counts
        hk1.merge(&hk2).unwrap();
        assert_eq!(hk1.count(&items[0]), 9, "Count should be sum of both instances");
        assert_eq!(hk1.count(&items[1]), 3, "Count should be preserved");
        assert_eq!(hk1.count(&items[2]), 6, "Count should be preserved");
    }

    /// Tests merge with incompatible width
    #[test]
    fn test_merge_incompatible_width() {
        let mut hk1: TopK<Vec<u8>> = TopK::with_seed(3, 100, 5, 0.9, 12345);
        let hk2 = TopK::with_seed(3, 50, 5, 0.9, 12345);

        match hk1.merge(&hk2) {
            Err(HeavyKeeperError::IncompatibleWidth { self_width, other_width }) => {
                assert_eq!(self_width, 100, "Self width should be 100");
                assert_eq!(other_width, 50, "Other width should be 50");
            }
            _ => panic!("Expected Width error"),
        }
    }

    /// Tests merge with incompatible depth
    #[test]
    fn test_merge_incompatible_depth() {
        let mut hk1: TopK<Vec<u8>> = TopK::with_seed(3, 100, 5, 0.9, 12345);
        let hk2 = TopK::with_seed(3, 100, 4, 0.9, 12345);

        match hk1.merge(&hk2) {
            Err(HeavyKeeperError::IncompatibleDepth { self_depth, other_depth }) => {
                assert_eq!(self_depth, 5, "Self depth should be 5");
                assert_eq!(other_depth, 4, "Other depth should be 4");
            }
            _ => panic!("Expected Depth error"),
        }
    }

    /// Tests merge with overlapping items
    #[test]
    fn test_merge_with_overlapping_items() {
        let seed = 12345;
        let mut hk1 = TopK::with_seed(3, 100, 5, 0.9, seed);
        let mut hk2 = TopK::with_seed(3, 100, 5, 0.9, seed);

        let items = [
            b"common".to_vec(),
            b"unique1".to_vec(),
            b"unique2".to_vec(),
        ];

        // Add overlapping items
        hk1.add(&items[0], 5);
        hk2.add(&items[0], 5);

        hk1.add(&items[1], 1);
        hk2.add(&items[2], 1);

        // Merge and verify counts
        hk1.merge(&hk2).unwrap();
        assert_eq!(hk1.count(&items[0]), 10, "Common item count should be doubled");
        assert_eq!(hk1.count(&items[1]), 1, "Unique item count should be preserved");
        assert_eq!(hk1.count(&items[2]), 1, "Unique item count should be preserved");
    }

    #[test]
    fn test_decay_logic_with_mock_rng() {
        let mut mock_rng = MockRngCoreTrait::new();
        mock_rng.expect_next_u64()
            .times(1..) // Allow multiple calls
            .return_const(0u64);
        
        let topk = TopK::<Vec<u8>>::builder()
            .k(1)
            .width(1)
            .depth(1)
            .decay(0.9)
            .rng(mock_rng)
            .build()
            .unwrap();
        
        let item1 = b"item1".to_vec();
        let item2 = b"item2".to_vec(); // Different item to trigger decay
        
        // Add item1 with a very large count (beyond lookup table)
        let large_count = 9999u64;
        let mut topk = topk;
        
        // Overwrite decay thresholds to always trigger decay
        topk.decay_thresholds.iter_mut().for_each(|threshold| {
            *threshold = u64::MAX; // Always trigger decay
        });
        
        topk.add(&item1, large_count);
        
        // Verify the initial count
        assert_eq!(topk.count(&item1), large_count);
        
        // Add item2 multiple times to trigger decay on item1
        let decay_iterations = 1000;
        let mut last_count = topk.bucket_count(&item1);
        for _ in 0..decay_iterations {
            topk.add(&item2, 1);
            let new_count = topk.bucket_count(&item1);
            if new_count == 0 {
                // Item has been evicted
                assert!(!topk.query(&item1), "item1 should be evicted if count is zero");
                break;
            } else {
                assert!(new_count < last_count, "Bucket count should decrease with each decay");
                last_count = new_count;
            }
        }
    }

    #[test]
    fn test_decay_and_eviction() {
        let mut mock_rng = MockRngCoreTrait::new();
        mock_rng.expect_next_u64()
            .times(1..)
            .return_const(0u64);

        let topk = TopK::<Vec<u8>>::builder()
            .k(1)
            .width(1)
            .depth(1)
            .decay(0.9)
            .rng(mock_rng)
            .build()
            .unwrap();

        let mut topk = topk;

        // Overwrite decay thresholds to always trigger decay
        topk.decay_thresholds.iter_mut().for_each(|threshold| {
            *threshold = u64::MAX; // Always trigger decay
        });

        let item1 = b"item1".to_vec();
        let item2 = b"item2".to_vec();
        let start_count = 10;
        topk.add(&item1, start_count);
        assert_eq!(topk.count(&item1), start_count);
        
        // Print fingerprints
        let fp1 = crate::hash_composition::HashComposer::new(&topk.hasher, &item1).fingerprint();
        let fp2 = crate::hash_composition::HashComposer::new(&topk.hasher, &item2).fingerprint();
        println!("item1 fingerprint: {}", fp1);
        println!("item2 fingerprint: {}", fp2);
        
        println!("Initial state:");
        println!("  item1 count: {}", topk.count(&item1));
        println!("  item1 query: {}", topk.query(&item1));
        println!("  item2 count: {}", topk.count(&item2));
        println!("  item2 query: {}", topk.query(&item2));

        // Add item2 once to trigger decay on item1
        let before = topk.bucket_count(&item1);
        println!("Before adding item2: item1 bucket count = {}", before);
        topk.add(&item2, 1);
        let after = topk.bucket_count(&item1);
        println!("After adding item2: item1 bucket count = {}", after);
        
        println!("Final state:");
        println!("  item1 count: {}", topk.count(&item1));
        println!("  item1 query: {}", topk.query(&item1));
        println!("  item2 count: {}", topk.count(&item2));
        println!("  item2 query: {}", topk.query(&item2));
        
        // After the first decay, item1's bucket count should be decremented by 1
        assert_eq!(after, before - 1, "Bucket count should decrement by 1 after first decay");
        
        // Since the count is still > 0, item1 should still be in the bucket
        // and item2 should not have taken over the bucket
        assert!(topk.query(&item1), "Item1 should still be in the bucket after decay");
        assert_eq!(topk.bucket_count(&item1), 9, "Item1 bucket count should be 9 after decay");
        assert!(!topk.query(&item2), "Item2 should not be in the bucket yet");
        assert_eq!(topk.bucket_count(&item2), 0, "Item2 bucket count should still be 0");
        
        // Now add item2 again to trigger another decay
        topk.add(&item2, 1);
        let final_count = topk.bucket_count(&item1);
        println!("After second decay: item1 bucket count = {}", final_count);
        
        // After multiple decays, item1 should eventually be evicted
        // For this test, we'll just verify the count continues to decrease
        assert!(final_count < 9, "Item1 bucket count should continue to decrease with more decays");
    }

    #[test]
    fn test_builder_missing_fields() {
        // Test missing k
        let result = TopK::<Vec<u8>>::builder()
            .width(100)
            .depth(5)
            .decay(0.9)
            .build();
        assert!(matches!(result, Err(BuilderError::MissingField { field }) if field == "k"));

        // Test missing width
        let result = TopK::<Vec<u8>>::builder()
            .k(10)
            .depth(5)
            .decay(0.9)
            .build();
        assert!(matches!(result, Err(BuilderError::MissingField { field }) if field == "width"));

        // Test missing depth
        let result = TopK::<Vec<u8>>::builder()
            .k(10)
            .width(100)
            .decay(0.9)
            .build();
        assert!(matches!(result, Err(BuilderError::MissingField { field }) if field == "depth"));

        // Test missing decay
        let result = TopK::<Vec<u8>>::builder()
            .k(10)
            .width(100)
            .depth(5)
            .build();
        assert!(matches!(result, Err(BuilderError::MissingField { field }) if field == "decay"));
    }

    #[test]
    fn test_send_sync_issue() {
        use std::sync::{Arc, Mutex};
        use std::collections::HashMap;
        use std::thread;
        
        type Id = String;
        type IdTopK = Arc<Mutex<HashMap<Id, TopK<String>>>>;
        
        let topk_map: IdTopK = Arc::new(Mutex::new(HashMap::new()));
        
        thread::spawn(move || {
            let mut map = topk_map.lock().unwrap();
            let topk = TopK::new(10, 100, 5, 0.9);
            map.insert("test".to_string(), topk);
        });
    }

    /// Tests that `add` accepts borrowed values (e.g., &str and &[u8])
    #[test]
    fn test_borrow() {
        let mut topk: TopK<String> = TopK::new(10, 100, 5, 0.9);
        let item: &str = "foo";
        topk.add(item, 1);
        assert!(topk.query(item));
        assert_eq!(topk.count(item), 1);

        let mut topk: TopK<Vec<u8>> = TopK::new(10, 100, 5, 0.9);
        let item: &[u8] = b"foo";
        topk.add(item, 1);
        assert!(topk.query(item));
        assert_eq!(topk.count(item), 1);
    }

    /// Tests that decay probability scaling uses the full u64 range
    ///
    /// With decay = 1.0 and RNG always returning exactly 2^63, the old
    /// implementation (which used threshold = 2^63) would never decay
    /// because `rand < threshold` was false for rand = 2^63.
    ///
    /// After the fix (threshold = u64::MAX), the same condition is true,
    /// so the bucket is always decayed and replaced as expected for
    /// probability 1.0.
    #[test]
    fn test_decay_probability_scaling_fix() {
        let mut mock_rng = MockRngCoreTrait::new();
        // Always return value exactly at the old threshold: 2^63.
        mock_rng
            .expect_next_u64()
            .times(1..) // Allow multiple calls
            .return_const(1u64 << 63);

        let mut topk = TopK::<Vec<u8>>::builder()
            .k(1)
            .width(1)
            .depth(1)
            .decay(1.0)
            .rng(mock_rng)
            .build()
            .unwrap();

        let item1 = b"item1".to_vec();
        let item2 = b"item2".to_vec();

        // First insert item1: single bucket becomes (fp(item1), count = 1).
        topk.add(&item1, 1);
        assert_eq!(topk.bucket_count(&item1), 1);
        assert_eq!(topk.bucket_count(&item2), 0);

        // Now insert item2, which collides into the same bucket and should
        // *always* trigger decay when decay_factor == 1.0.
        topk.add(&item2, 1);

        // With correct scaling, item1 is fully decayed and replaced by item2.
        assert_eq!(topk.bucket_count(&item1), 0);
        assert_eq!(topk.bucket_count(&item2), 1);
    }
}

