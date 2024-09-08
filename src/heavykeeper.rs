use ahash::RandomState;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
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

pub struct TopK<T: Eq + Hash + Clone + Debug> {
    top_items: usize,
    width: usize,
    depth: usize,
    decay_thresholds: Vec<u32>,
    buckets: Vec<Vec<Bucket>>,
    items: Vec<(T, u32)>,
    item_indices: HashMap<T, usize>,
    hasher: RandomState,
    random: SmallRng,
}

impl<T: Eq + Hash + Clone + Debug> TopK<T> {
    pub fn new(k: usize, width: usize, depth: usize, decay: f64) -> Self {
        let decay_thresholds = precompute_decay_thresholds(decay, DECAY_LOOKUP_SIZE);
        let buckets = vec![vec![Bucket::default(); width]; depth];
        let items = Vec::with_capacity(k);
        let item_indices = HashMap::with_capacity(k);

        TopK {
            top_items: k,
            width,
            depth,
            decay_thresholds,
            buckets,
            items,
            item_indices,
            hasher: RandomState::with_seeds(0, 0, 0, 0),
            random: SmallRng::from_entropy(),
        }
    }

    pub fn add(&mut self, item: T) {
        let item_fingerprint = self.hash(&item);
        let mut max_count: u32 = 0;

        for i in 0..self.depth {
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
            } else {
                let decay_threshold = if (bucket.count as usize) < self.decay_thresholds.len() {
                    self.decay_thresholds[bucket.count as usize]
                } else {
                    self.decay_thresholds.last().cloned().unwrap_or_default()
                };
                let rand = self.random.gen::<u32>();
                if rand < decay_threshold {
                    bucket.count = bucket.count.saturating_sub(1);
                }
            }
        }

        self.update_priority_queue(item, max_count);
    }

    fn update_priority_queue(&mut self, item: T, new_count: u32) {
        if let Some(&index) = self.item_indices.get(&item) {
            // Update existing item
            let (_, count) = &mut self.items[index];
            if new_count > *count {
                *count = new_count;
                self.sift_down(index);
            }
        } else if self.items.len() < self.top_items {
            // Add new item to non-full list
            let index = self.items.len();
            self.items.push((item.clone(), new_count));
            self.item_indices.insert(item, index);
            self.sift_up(index);
        } else if new_count > self.items[0].1 {
            // Replace smallest item
            let old_item = std::mem::replace(&mut self.items[0].0, item.clone());
            self.items[0].1 = new_count;
            self.item_indices.remove(&old_item);
            self.item_indices.insert(item, 0);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.items[index].1 < self.items[parent].1 {
                self.items.swap(index, parent);
                self.item_indices.insert(self.items[index].0.clone(), index);
                self.item_indices.insert(self.items[parent].0.clone(), parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut smallest = index;

            if left < self.items.len() && self.items[left].1 < self.items[smallest].1 {
                smallest = left;
            }
            if right < self.items.len() && self.items[right].1 < self.items[smallest].1 {
                smallest = right;
            }

            if smallest != index {
                self.items.swap(index, smallest);
                self.item_indices.insert(self.items[index].0.clone(), index);
                self.item_indices.insert(self.items[smallest].0.clone(), smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

    pub fn query(&self, item: &T) -> bool {
        self.item_indices.contains_key(item)
    }

    pub fn count(&self, item: &T) -> u32 {
        self.item_indices
            .get(item)
            .map(|&index| self.items[index].1)
            .unwrap_or(0)
    }

    pub fn list(&self) -> Vec<Node<T>> {
        let mut nodes: Vec<_> = self.items
            .iter()
            .map(|(item, count)| Node { item: item.clone(), count: *count })
            .collect();
        nodes.sort_by(|a, b| b.count.cmp(&a.count));
        nodes
    }
    
    fn hash<B: Hash>(&self, item: B) -> u64 {
        self.hasher.hash_one(item)
    }
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
        assert_eq!(topk.buckets.len(), 5);
        assert_eq!(topk.buckets[0].len(), 100);
        assert_eq!(topk.items.len(), 0);
        assert_eq!(topk.item_indices.len(), 0);
    }

    #[test]
    fn test_query() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);
        topk.add("hello".as_bytes().to_vec());

        assert!(topk.query(&"hello".as_bytes().to_vec()));
        assert!(!topk.query(&"world".as_bytes().to_vec()));
    }

    #[test]
    fn test_count() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        for _ in 0..8 {
            topk.add("lashin".as_bytes().to_vec());
        }
        assert_eq!(topk.count(&"lashin".as_bytes().to_vec()), 8);
        assert_eq!(topk.count(&"पुष्पं अस्ति।".as_bytes().to_vec()), 0);

        for _ in 0..1337 {
            topk.add("ballynamoney".as_bytes().to_vec());
        }
        assert_eq!(topk.count(&"ballynamoney".as_bytes().to_vec()), 1337);
    }

    #[test]
    fn test_add_single_item() {
        let k = 1;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes().to_vec());

        let nodes = topk.list();

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].count, 1);
        assert_eq!(nodes[0].item, "hello".as_bytes().to_vec());
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
            topk.add("hello".as_bytes().to_vec());
        }

        // Add "world" 7 times
        for _ in 0..7 {
            topk.add("world".as_bytes().to_vec());
        }

        assert_eq!(topk.items.len(), k);

        let nodes = topk.list();

        assert_eq!(nodes.len(), 2);
        
        // Sort the nodes by count in descending order
        let mut sorted_nodes = nodes.clone();
        sorted_nodes.sort_by(|a, b| b.count.cmp(&a.count));
        // Check that both "hello" and "world" are in the top items
        let top_items: Vec<&[u8]> = sorted_nodes.iter().map(|node| node.item.as_slice()).collect();
        assert!(top_items.contains(&b"hello".as_slice()));
        assert!(top_items.contains(&b"world".as_slice()));

        // Check that both counts are 7
        assert_eq!(sorted_nodes[0].count, 7);
        assert_eq!(sorted_nodes[1].count, 7);
    }

    #[test]
    fn test_add_more_items_than_capacity() {
        let k = 2;
        let width = 100;
        let depth = 5;
        let decay = 0.9;

        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes().to_vec());
        topk.add("world".as_bytes().to_vec());
        topk.add("ballynamoney".as_bytes().to_vec());
        topk.add("lane".as_bytes().to_vec());

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

        topk.add("hello".as_bytes().to_vec());
        topk.add("world".as_bytes().to_vec());
        topk.add("ballynamoney".as_bytes().to_vec());
        topk.add("lane".as_bytes().to_vec());
        topk.add("pear tree".as_bytes().to_vec());

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
            let item = format!("item{}", i).into_bytes();
            let frequency = i + 1; // Ensure varied frequencies
            items_with_frequencies.push((item, frequency));
        }

        // Add items based on their frequencies
        for (item, frequency) in items_with_frequencies.iter() {
            for _ in 0..*frequency {
                topk.add(item.clone());
            }
        }

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.items.len(),
            k,
            "Min-heap does not contain the top-k items"
        );

        // Verify the min-heap contains the correct top-k items based on frequency
        let top_items = topk.list();

        let expected_top_items: Vec<Vec<u8>> = (90..100)
            .map(|i| format!("item{}", i).into_bytes())
            .collect();

        println!("Expected top items: {:?}", expected_top_items);
        println!("Actual top items: {:?}", top_items);

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {:?} to be in the top-k items",
                String::from_utf8_lossy(expected_item)
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

        // Generate 3 unique items with varied addition frequencies
        for i in 0..3 {
            let item = format!("item{}", i).into_bytes();
            for _ in 0..(i + 1) {
                topk.add(item.clone());
            }
        }

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.items.len(),
            k,
            "Min-heap does not contain the top-k items"
        );

        // Verify the min-heap contains the correct top-k items based on frequency
        let top_items = topk.list();

        let expected_top_items: Vec<Vec<u8>> = (1..3)
            .map(|i| format!("item{}", i).into_bytes())
            .collect();

        println!("Expected top items: {:?}", expected_top_items);
        println!("Actual top items: {:?}", top_items);

        for expected_item in expected_top_items.iter() {
            assert!(
                top_items.iter().any(|node| &node.item == expected_item),
                "Expected item {:?} to be in the top-k items",
                String::from_utf8_lossy(expected_item)
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

        let item = "test_item".as_bytes().to_vec();
        let num_additions = 1000;

        // Add the same item a large number of times
        for _ in 0..num_additions {
            topk.add(item.clone());
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

        let item1 = "item1".as_bytes().to_vec();
        let item2 = "item2".as_bytes().to_vec();
        let num_additions_item1 = 500;
        let num_additions_item2 = 499; // One less than item1

        // Add item1 multiple times
        for _ in 0..num_additions_item1 {
            topk.add(item1.clone());
        }

        // Add item2 one less time than item1
        for _ in 0..num_additions_item2 {
            topk.add(item2.clone());
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
        let frequent_item = "frequent_item".as_bytes().to_vec();
        let less_frequent_item = "less_frequent_item".as_bytes().to_vec();
        let frequent_additions = 100;
        let less_frequent_additions = 50;

        // Add the frequent item many times to ensure it hits the decay condition when appropriate
        for _ in 0..frequent_additions {
            topk.add(frequent_item.clone());
        }

        // Triggering decay implicitly through normal use
        for _ in 0..less_frequent_additions {
            topk.add(less_frequent_item.clone());
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
        let item = "new_flow".as_bytes().to_vec();

        // Adding the new item
        topk.add(item.clone());

        // Verify that the item has been added with an initial count of 1
        let item_hash = topk.hash(&item);
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
            let item = format!("item{}", i).into_bytes();
            for _ in 0..frequency {
                topk.add(item.clone());
            }
        }

        // Verify the min-heap has exactly k items
        assert_eq!(
            topk.items.len(),
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
}
