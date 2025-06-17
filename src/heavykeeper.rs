use crate::hash_composition::HashComposer;
use crate::priority_queue::TopKQueue;
use ahash::RandomState;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::clone::Clone;
use std::fmt::Debug;
use std::hash::Hash;
use thiserror::Error;

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
    IncompatibleDecay { self_decay: f64, other_decay: f64 },

    #[error("Incompatible top_items: self ({self_items}) != other ({other_items})")]
    IncompatibleTopItems {
        self_items: usize,
        other_items: usize,
    },
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
    random: SmallRng,
}

fn precompute_decay_thresholds(decay: f64, num_entries: usize) -> Vec<u64> {
    let mut thresholds = Vec::with_capacity(num_entries);
    for count in 0..num_entries {
        let decay_factor = decay.powf(count as f64);
        let threshold = (decay_factor * (1u64 << 63) as f64) as u64;
        thresholds.push(threshold);
    }
    thresholds
}

impl<T: Ord + Clone + Hash + Debug> TopK<T> {
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

    pub fn with_hasher(
        k: usize,
        width: usize,
        depth: usize,
        decay: f64,
        hasher: RandomState,
    ) -> Self {
        // Pre-allocate with capacity to avoid resizing
        let mut buckets = Vec::with_capacity(depth);
        for _ in 0..depth {
            buckets.push(vec![
                Bucket {
                    fingerprint: 0,
                    count: 0
                };
                width
            ]);
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
            random: SmallRng::seed_from_u64(0),
        }
    }

    pub fn query(&self, item: &T) -> bool {
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

    pub fn count(&self, item: &T) -> u64 {
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

    pub fn add(&mut self, item: &T) {
        let mut composer = HashComposer::new(&self.hasher, item);
        let mut max_count: u64 = 0;

        for i in 0..self.depth {
            let bucket_idx = composer.next_bucket(self.width as u64, i);
            let bucket = &mut self.buckets[i][bucket_idx];

            let matches = bucket.fingerprint == composer.fingerprint();
            let empty = bucket.count == 0u64;

            if matches || empty {
                bucket.fingerprint = composer.fingerprint();
                bucket.count += 1;
                max_count = std::cmp::max(max_count, bucket.count);
            } else {
                let count_idx = bucket.count as usize;
                let decay_threshold = if count_idx < self.decay_thresholds.len() {
                    self.decay_thresholds[count_idx]
                } else {
                    self.decay_thresholds.last().cloned().unwrap_or_default()
                };
                let rand = self.random.random::<u64>();
                if rand < decay_threshold {
                    bucket.count = bucket.count.saturating_sub(1);
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
        self.priority_queue.upsert(item.clone(), max_count);
    }

    pub fn list(&self) -> Vec<Node<T>> {
        let mut nodes = self
            .priority_queue
            .iter()
            .map(|(item, count)| Node {
                item: item.clone(),
                count,
            })
            .collect::<Vec<_>>();
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
        let mut nodes = self
            .priority_queue
            .iter()
            .map(|(item, count)| Node {
                item: item.clone(),
                count,
            })
            .collect::<Vec<_>>();

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

#[cfg(test)]
mod tests {
    use super::*;

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
        topk.add(&present);

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
        for _ in 0..8 {
            topk.add(&item1);
        }
        assert_eq!(
            topk.count(&item1),
            8,
            "Count should match number of additions"
        );

        // Verify count for non-existent item
        assert_eq!(
            topk.count(&item3),
            0,
            "Non-existent item should have count 0"
        );

        // Add second item many times
        for _ in 0..1337 {
            topk.add(&item2);
        }
        assert_eq!(
            topk.count(&item2),
            1337,
            "Count should match number of additions"
        );
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
        topk.add(&p);
        topk.add(&emoji);
        topk.add(&mixed);

        // Verify presence
        assert!(topk.query(&p), "text should be found");
        assert!(topk.query(&emoji), "Emoji should be found");
        assert!(topk.query(&mixed), "Mixed content should be found");

        // Verify counts
        assert_eq!(topk.count(&p), 1, "text count should be 1");
        assert_eq!(topk.count(&emoji), 1, "Emoji count should be 1");
        assert_eq!(topk.count(&mixed), 1, "Mixed content count should be 1");

        // Add more occurrences
        for _ in 0..4 {
            topk.add(&p);
        }
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
        topk.add(&item);

        let nodes = topk.list();
        assert_eq!(nodes.len(), 1, "Should have exactly one item");
        assert_eq!(nodes[0].count, 1, "Count should be 1");
        assert_eq!(nodes[0].item, item, "Item should match");
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
        for _ in 0..7 {
            topk.add(&item1);
            topk.add(&item2);
        }

        assert_eq!(topk.priority_queue.len(), k, "Should have exactly k items");

        let nodes = topk
            .priority_queue
            .iter()
            .map(|(item, count)| Node {
                item: item.clone(),
                count,
            })
            .collect::<Vec<_>>();

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
            topk.add(item);
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
            topk.add(item);
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
        let width = 1000;
        let depth = 10;
        let decay = 0.95;

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
                topk.add(&item_bytes);
            }
        }

        // Verify the priority queue has exactly k items
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Priority queue should contain exactly k items"
        );

        // Verify the top-k items are correct
        let top_items = topk
            .priority_queue
            .iter()
            .map(|(item, count)| Node {
                item: std::str::from_utf8(item).unwrap().to_string().into_bytes(),
                count,
            })
            .collect::<Vec<_>>();

        let expected_top_items = (90..100)
            .map(|i| format!("item{}", i).into_bytes())
            .collect::<Vec<_>>();

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {} to be in top-k",
                std::str::from_utf8(expected_item).unwrap()
            );
        }
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
        for _ in 0..num_additions {
            topk.add(&item);
        }

        assert_eq!(
            topk.count(&item),
            num_additions,
            "Count should match number of additions"
        );
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
        for _ in 0..num_additions_item1 {
            topk.add(&item1);
        }
        for _ in 0..num_additions_item2 {
            topk.add(&item2);
        }

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
        topk.add(&item);

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
            for _ in 0..frequency {
                topk.add(&item_bytes);
            }
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
                node.count, frequency,
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
            for _ in 0..(i + 1) {
                topk.add(&item_bytes);
            }
        }

        // Verify priority queue size
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Priority queue should contain exactly k items"
        );

        // Verify top-k items
        let top_items = topk
            .priority_queue
            .iter()
            .map(|(item, count)| Node {
                item: std::str::from_utf8(item).unwrap().to_string().into_bytes(),
                count,
            })
            .collect::<Vec<_>>();

        let expected_top_items = (1..3)
            .map(|i| format!("item{}", i).into_bytes())
            .collect::<Vec<_>>();

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
        topk.add(&items[0]);
        topk.add(&items[1]);
        for _ in 0..2 {
            topk.add(&items[2]);
        }
        for _ in 0..5 {
            topk.add(&items[3]);
        }

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

        let items = [b"item1".to_vec(), b"item2".to_vec(), b"item3".to_vec()];

        // Add items to first instance
        for _ in 0..5 {
            hk1.add(&items[0]);
        }
        for _ in 0..3 {
            hk1.add(&items[1]);
        }

        // Add items to second instance
        for _ in 0..4 {
            hk2.add(&items[0]);
        }
        for _ in 0..6 {
            hk2.add(&items[2]);
        }

        // Merge and verify counts
        hk1.merge(&hk2).unwrap();
        assert_eq!(
            hk1.count(&items[0]),
            9,
            "Count should be sum of both instances"
        );
        assert_eq!(hk1.count(&items[1]), 3, "Count should be preserved");
        assert_eq!(hk1.count(&items[2]), 6, "Count should be preserved");
    }

    /// Tests merge with incompatible width
    #[test]
    fn test_merge_incompatible_width() {
        let mut hk1: TopK<Vec<u8>> = TopK::with_seed(3, 100, 5, 0.9, 12345);
        let hk2 = TopK::with_seed(3, 50, 5, 0.9, 12345);

        match hk1.merge(&hk2) {
            Err(HeavyKeeperError::IncompatibleWidth {
                self_width,
                other_width,
            }) => {
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
            Err(HeavyKeeperError::IncompatibleDepth {
                self_depth,
                other_depth,
            }) => {
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

        let items = [b"common".to_vec(), b"unique1".to_vec(), b"unique2".to_vec()];

        // Add overlapping items
        for _ in 0..5 {
            hk1.add(&items[0]);
            hk2.add(&items[0]);
        }

        hk1.add(&items[1]);
        hk2.add(&items[2]);

        // Merge and verify counts
        hk1.merge(&hk2).unwrap();
        assert_eq!(
            hk1.count(&items[0]),
            10,
            "Common item count should be doubled"
        );
        assert_eq!(
            hk1.count(&items[1]),
            1,
            "Unique item count should be preserved"
        );
        assert_eq!(
            hk1.count(&items[2]),
            1,
            "Unique item count should be preserved"
        );
    }
}
