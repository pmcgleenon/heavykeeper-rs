use ahash::RandomState;
use std::clone::Clone;
use std::cmp::Reverse;

use priority_queue::PriorityQueue;
use std::fmt::Debug;
use std::hash::{Hash};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

const DECAY_LOOKUP_SIZE: usize = 1024;

#[derive(Default, Clone, Debug)]
struct Bucket {
    fingerprint: u64,
    count: u32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node<T> {
    pub item: T,
    pub count: u32,
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

pub struct TopK<T: Ord + Clone + Hash + Debug> {
    top_items: usize,
    width: usize,
    depth: usize,
    decay: f64,
    decay_thresholds: Vec<u32>,
    buckets: Vec<Vec<Bucket>>,
    priority_queue: PriorityQueue<T, Reverse<u32>>,
    hasher: RandomState,
    random: SmallRng,
}

fn precompute_decay_thresholds(decay: f64, num_entries: usize) -> Vec<u32> {
    let mut thresholds = Vec::with_capacity(num_entries);
    for count in 0..num_entries {
        let decay_factor = decay.powf(count as f64);
        let threshold = (decay_factor * (1u32 << 31) as f64) as u32;
        thresholds.push(threshold);
    }
    thresholds
}


impl<T: Ord + Clone  + Hash + Debug> TopK<T> {
    pub fn new(k: usize, width: usize, depth: usize, decay: f64) -> Self {
        let decay_thresholds = precompute_decay_thresholds(decay, DECAY_LOOKUP_SIZE);
        let buckets = vec![vec![Bucket::default(); width]; depth];

        let ahash_hasher = RandomState::with_seeds(0, 0, 0, 0);
        let priority_queue : PriorityQueue<T, Reverse<u32>> = PriorityQueue::with_capacity(k);

        TopK {
            top_items: k,
            width,
            depth,
            decay,
            decay_thresholds,
            buckets,
            priority_queue,
            hasher: ahash_hasher,
            random: SmallRng::from_entropy(),
        }
    }

    pub fn query(&self, item: &T) -> bool {
        self.priority_queue.get(item).is_some()
    }

    pub fn count(&self, item: &T) -> u32 {
        let node = self.priority_queue.get(item);

        // return the count
        match node {
            Some(count) => count.1.0,
            None => 0,
        }
    }

    fn hash<B: Hash>(&mut self, item: B) -> u64 {
        // let mut hasher = AHasher::default();
        // item.hash(&mut hasher);
        // hasher.finish()

        self.hasher.hash_one(item)
    }


    // TODO replace this with iterator
    pub fn list(&self) -> Vec<Node<T>> {
        let mut nodes = self.priority_queue.iter().map(|(item, count)| Node {
            item: item.clone(),
            count: count.0,
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
            count: count.0,
        }).collect::<Vec<_>>();

        nodes.sort();
        for node in nodes {
            println!("Node - Item: {:?}, Count: {}", node.item, node.count);
        }
    }

    pub fn add(&mut self, item: T) {
        let item_fingerprint = self.hash(&item);

        let mut max_count: u32 = 0;

        for i in 0..self.depth {
            // Combine item fingerprint and depth index to generate a unique bucket index
            let combined = (item_fingerprint, i);
            let bucket_idx = self.hash(combined) % self.width as u64;
            let bucket_idx = bucket_idx as usize;
            let bucket = &mut self.buckets[i][bucket_idx];

            let matches = bucket.fingerprint == item_fingerprint;
            let empty = bucket.count == 0;

            if matches || empty {
                bucket.fingerprint = item_fingerprint;
                bucket.count += 1;
                max_count = std::cmp::max(max_count, bucket.count);
            }
            else {
                // Use the precomputed decay threshold based on the bucket count
                let decay_threshold = if (bucket.count as usize) < self.decay_thresholds.len() {
                    self.decay_thresholds[bucket.count as usize]
                } else {
                    self.decay_thresholds.last().cloned().unwrap_or_default()
                };
                // Apply bitwise decay based on the decay threshold
                let rand = self.random.gen::<u32>();
                if rand < decay_threshold {
                    bucket.count = bucket.count.saturating_sub(1);
                }
            }
        }

        let space_in_heap = self.priority_queue.len() < self.top_items;
        let curr_min = self.priority_queue.peek().map_or(0, |v| v.1 .0);
        // skip if max_count is less than the smallest count in the min-heap
        if max_count < curr_min && !space_in_heap {
            return;
        }

        let new_max = max_count >= curr_min;

        if self.priority_queue.get(&item).is_some() {
            self.priority_queue.change_priority(&item, Reverse(max_count));
        }  else if space_in_heap || new_max {
            if self.priority_queue.len() >= self.top_items {
                self.priority_queue.pop();
            }
            self.priority_queue.push(item, Reverse(max_count));
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

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

    #[test]
    fn test_query() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);
        topk.add("hello".as_bytes());

        assert!(topk.query(&"hello".as_bytes()));
        assert!(!topk.query(&"world".as_bytes()));
    }

    #[test]
    fn test_count() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        for _ in 0..8 {
            topk.add("lashin".as_bytes());
        }
        assert_eq!(topk.count(&"lashin".as_bytes()), 8);
        assert_eq!(topk.count(&"पुष्पं अस्ति।".as_bytes()), 0);

        for _ in 0..1337 {
            topk.add("ballynamoney".as_bytes());
        }
        assert_eq!(topk.count(&"ballynamoney".as_bytes()), 1337);
    }

    #[test]
    fn test_add_single_item() {
        let k = 1;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes());

        let nodes = topk.list();
        //let nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].count, 1);
        assert_eq!(nodes[0].item, "hello".as_bytes());
    }

    #[test]
    fn test_add_duplicate_items() {
        let k = 2; // 2 most frequent items
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk = TopK::new(k, width, depth, decay);

        // Add "hello" 7 times
        for _ in 0..7 {
            topk.add("hello".as_bytes());
        }

        // Add "world" 7 times
        for _ in 0..7 {
            topk.add("world".as_bytes());
        }

        assert_eq!(topk.priority_queue.len(), k); // Assertion for min_heap length

        let nodes = topk.priority_queue.iter().map(|(item, count)| Node {
            item: *item,
            count: count.0,
        }).collect::<Vec<_>>();

        assert_eq!(nodes.len(), 2); // Assertion for nodes length
        assert_eq!(nodes[0].count, 7);
        assert_eq!(nodes[0].item, "hello".as_bytes());
        assert_eq!(nodes[1].count, 7);
        assert_eq!(nodes[1].item, "world".as_bytes());
    }

    #[test]
    fn test_add_more_items_than_capacity() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes());
        topk.add("world".as_bytes());
        topk.add("ballynamoney".as_bytes());
        topk.add("lane".as_bytes());

        let nodes = topk.list();

        assert_eq!(nodes.len(), 2);
        let mut counts = nodes.iter().map(|node| node.count).collect::<Vec<_>>();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 1]);
    }

    #[test]
    fn test_add_with_different_decay() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.5; // Lower decay value

        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes());
        topk.add("world".as_bytes());
        topk.add("ballynamoney".as_bytes());
        topk.add("lane".as_bytes());
        topk.add("pear tree".as_bytes());

        let nodes = topk.list();

        assert_eq!(nodes.len(), 2);
        let mut counts = nodes.iter().map(|node| node.count).collect::<Vec<_>>();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 1]);
    }

    #[test]
    fn test_add_empty_input() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        //let nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();
        let nodes = topk.list();

        assert_eq!(nodes.len(), 0);
    }

    #[test]
    fn test_add_varied_input() {
        let k = 10; // We want to track the top-10 items
        let width = 1000;
        let depth = 10;
        let decay = 0.95;

        let mut topk = TopK::new(k, width, depth, decay);

        // Generate 100 unique items with varied addition frequencies
        let mut items_with_frequencies = Vec::new();
        for i in 0..100 {
            let item = format!("item{}", i);
            let frequency = i + 1; // Ensure varied frequencies
            items_with_frequencies.push((item, frequency));
        }

        // Add items based on their frequencies
        for (item, frequency) in items_with_frequencies.iter() {
            for _ in 0..*frequency {
                topk.add(item.as_bytes().to_vec());
            }
        }

        topk.debug();

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Min-heap does not contain the top-k items"
        );

        // Verify the min-heap contains the correct top-k items based on frequency
        // The top-k items should be the last k items added due to their higher frequencies
        let top_items = topk.priority_queue.iter().map(|(item, count)| Node {
            item: std::str::from_utf8(item).unwrap().to_string(), // Convert byte array back to string
            count: count.0,
        }).collect::<Vec<_>>();

        let expected_top_items = (90..100).map(|i| format!("item{}", i)).collect::<Vec<_>>();
        //let expected_top_items = items_with_frequencies.iter().skip(90).map(|(item, frequency)| (std::str::from_utf8(item.as_ref()).unwrap().to_string(), *frequency)).collect::<Vec<_>>();

        println!("Expected top items: {:?}", expected_top_items);
        println!("Actual top items: {:?}", top_items);

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {} to be in the top-k items",
                expected_item
            );
        }
    }

    #[test]
    fn test_large_number_of_duplicates() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk = TopK::new(k, width, depth, decay);

        let item = "test_item".as_bytes();
        let num_additions = 1000;

        // Add the same item a large number of times
        for _ in 0..num_additions {
            topk.add(item);
        }

        // Query the count for the item
        let count = topk.count(&item);

        assert_eq!(count, num_additions);
    }

    #[test]
    fn test_multiple_distinct_items() {
        let k = 2; // We want to track the top-2 items
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk = TopK::new(k, width, depth, decay);

        let item1 = "item1".as_bytes();
        let item2 = "item2".as_bytes();
        let num_additions_item1 = 500;
        let num_additions_item2 = 499; // One less than item1

        // Add item1 multiple times
        for _ in 0..num_additions_item1 {
            topk.add(item1);
        }

        // Add item2 one less time than item1
        for _ in 0..num_additions_item2 {
            topk.add(item2);
        }

        // Query the count for both items
        let count_item1 = topk.count(&item1);
        let count_item2 = topk.count(&item2);

        // Assert that both items' counts are as expected
        assert_eq!(
            count_item1, num_additions_item1,
            "The count for item1 does not match the expected value."
        );
        assert_eq!(
            count_item2, num_additions_item2,
            "The count for item2 does not match the expected value."
        );

        // Additionally, check if both items are in the top-K list
        assert!(topk.query(&item1), "item1 should be in the top-K list.");
        assert!(topk.query(&item2), "item2 should be in the top-K list.");
    }

    #[test]
    fn test_decay_through_normal_use() {
        let k = 2; // We want to track the top-2 items
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        // Define items and their addition counts
        let frequent_item = "frequent_item".as_bytes();
        let less_frequent_item = "less_frequent_item".as_bytes();
        let frequent_additions = 100;
        let less_frequent_additions = 50;

        // Add the frequent item many times to ensure it hits the decay condition when appropriate
        for _ in 0..frequent_additions {
            topk.add(frequent_item);
        }

        // Triggering decay implicitly through normal use
        for _ in 0..less_frequent_additions {
            topk.add(less_frequent_item);
        }

        // Verify the counts to ensure decay has been applied
        // This will depend on knowing the expected outcome after decay
        // For this example, let's check if the frequent item's count is still correctly leading
        let count_frequent = topk.count(&frequent_item);
        let count_less_frequent = topk.count(&less_frequent_item);

        assert!(
            count_frequent > count_less_frequent,
            "Decay does not seem to have been applied correctly."
        );
    }

    #[test]
    fn test_insertion_into_empty_buckets() {
        let k = 5; // Size of the min-heap
        let width = 10; // Width of each row in the buckets
        let depth = 4; // Depth of the bucket array
        let decay = 0.5; // Decay factor
        let mut topk = TopK::new(k, width, depth, decay);

        // New item expected to be inserted into an empty bucket
        let item = "new_flow".as_bytes();

        // Adding the new item
        topk.add(item);

        // Verify that the item has been added with an initial count of 1
        let item_hash = topk.hash(item);
        assert!(
            topk.buckets.iter().any(|row| row
                .iter()
                .any(|bucket| bucket.fingerprint == item_hash && bucket.count == 1)),
            "The item was not inserted into an empty bucket correctly."
        );

        // verify that the item is in the min-heap
        assert!(
            topk.query(&item),
            "The item was not inserted into the min-heap correctly."
        );
    }

    #[test]
    fn test_add_identical_frequencies() {
        let k = 10;
        let width = 1000;
        let depth = 10;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Generate 100 unique items with the same frequency
        let frequency = 5;
        for i in 0..100 {
            let item = format!("item{}", i);
            for _ in 0..frequency {
                topk.add(item.as_bytes().to_vec());
            }
        }

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Min-heap does not contain the top-k items"
        );

        // Since all items have the same frequency, we just check the count
        for node in topk.list() {
            assert_eq!(
                node.count, frequency,
                "All items should have the same frequency"
            );
        }
    }

    #[test]
    fn test_small_k_value2() {
        let k = 2; // Smaller k value
        let width = 1000;
        let depth = 10;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Generate 100 unique items with varied addition frequencies
        for i in 0..3 {
            let item = format!("item{}", i);
            for _ in 0..(i + 1) {
                topk.add(item.as_bytes().to_vec());
            }
        }

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.priority_queue.len(),
            k,
            "Min-heap does not contain the top-k items"
        );

        // Verify the min-heap contains the correct top-k items based on frequency
        let top_items = topk.priority_queue.iter().map(|(item, count)| Node {
            item: std::str::from_utf8(item).unwrap().to_string(), // Convert byte array back to string
            count: count.0,
        }).collect::<Vec<_>>();

        let expected_top_items = (1..3).map(|i| format!("item{}", i)).collect::<Vec<_>>();

        println!("Expected top items: {:?}", expected_top_items);
        println!("Actual top items: {:?}", top_items);

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {} to be in the top-k items",
                expected_item
            );
        }
    }
}
