use std::clone::Clone;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use ahash::AHasher;
use rand::{random};

#[derive(Default, Clone, Debug)]
struct Bucket {
    fingerprint: u64,
    count: u64,
}

impl Bucket {
    fn new() -> Self {
        Bucket {
            fingerprint: 0,
            count: 0,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Node<T> {
    pub count: u64,
    pub item: T,
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

pub struct TopK<T: Debug> {
    k: usize,
    width: usize,
    depth: usize,
    decay: f64,
    buckets: Vec<Vec<Bucket>>,
    min_heap: BinaryHeap<Node<T>>,
    item_counts: std::collections::HashMap<T, u64>,
}



    impl<T: Ord + Clone + Eq + PartialEq + AsRef<[u8]> + Hash + Debug> TopK<T> {

        pub fn new(k: usize, width: usize, depth: usize, decay: f64) -> Self {
            let mut buckets = Vec::with_capacity(depth);
            for _ in 0..depth {
                let mut row = Vec::with_capacity(width);
                for _ in 0..width {
                    row.push(Bucket::new());
                }
                buckets.push(row);
            }

            TopK {
                k,
                width,
                depth,
                decay,
                buckets,
                min_heap: BinaryHeap::with_capacity(k),
                item_counts: std::collections::HashMap::new(),
            }
        }


    pub fn query(&self, item: &T) -> bool {
        self.min_heap.iter().any(|node| &node.item == item)
    }

    pub fn count(&self, item: &T) -> Option<u64> {
        self.min_heap
            .iter()
            .find(|node| &node.item == item)
            .map(|node| node.count)
    }

    fn hash<B: Hash>(&self, item: B) -> u64 {
        let mut hasher = AHasher::default();
        item.hash(&mut hasher);
        hasher.finish()
    }

    pub fn list(&self) -> Vec<Node<T>> {
        let mut nodes = self.min_heap.iter().cloned().collect::<Vec<_>>();
        nodes.sort();
        nodes
    }
        pub fn debug(&self) {
            println!("k: {}", self.k);
            println!("width: {}", self.width);
            println!("depth: {}", self.depth);
            println!("decay: {}", self.decay);
            let mut buckets: Vec<(&Bucket, usize, usize)> = self.buckets.iter().enumerate()
                .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, bucket)| (bucket, i, j)))
                .filter(|(bucket, _, _)| bucket.count != 0)
                .collect();
            buckets.sort_by(|a, b| b.0.count.cmp(&a.0.count));
            for (bucket, i, j) in buckets {
                println!("Bucket at row {}, column {}: {:?}", i, j, bucket);
            }
            println!("min_heap: ");
            let mut min_heap: Vec<&Node<T>> = self.min_heap.iter().collect();
            min_heap.sort_by(|a, b| b.count.cmp(&a.count));
            for node in min_heap {
                let item_str = String::from_utf8_lossy(node.item.as_ref());
                println!("Node - Item: {}, Count: {}", item_str, node.count);
            }
            println!("item_counts: ");
            let mut item_counts: Vec<(&T, &u64)> = self.item_counts.iter().collect();
            item_counts.sort_by(|a, b| b.1.cmp(a.1));
            for (item, count) in item_counts {
                let item_str = String::from_utf8_lossy(item.as_ref());
                println!("{}: {}", item_str, count);
            }
        }

        pub fn add(&mut self, item: T) {
            let item_fingerprint = self.hash(&item);

            let mut max_count = 0;
            let heap_min = self.min_heap.peek().map(|node| node.count).unwrap_or_default();

            for i in 0..self.depth {
                // Combine item fingerprint and depth index to generate a unique bucket index
                let combined = (item_fingerprint, i);
                let bucket_idx = self.hash(&combined) % self.width as u64;
                let bucket_idx = bucket_idx as usize;
                let bucket = &mut self.buckets[i][bucket_idx];

                if bucket.fingerprint == item_fingerprint || bucket.count == 0 {
                    bucket.fingerprint = item_fingerprint;
                    bucket.count += 1;
                    max_count = std::cmp::max(max_count, bucket.count);
                } else {
                    // Apply decay based on the algorithm's logic
                    let decay_factor = self.decay.powi(bucket.count as i32);
                    //if rand::random::<f64>() < decay_factor {
                    if random::<f64>() < decay_factor {
                        bucket.count -= 1;
                    }
                }
            }

            // Update item count in HashMap
            let count = self.item_counts.entry(item.clone()).or_insert(0);
            *count = max_count;

            // Update the min_heap
            let mut found = false;
            let mut nodes = self.min_heap.drain().collect::<Vec<_>>();
            for node in nodes.iter_mut() {
                if node.item == item {
                    node.count = max_count; // Update count
                    found = true;
                    break;
                }
            }

            // Reinsert nodes back into the heap
            for node in nodes {
                self.min_heap.push(node);
            }

            if !found && (self.min_heap.len() < self.k || max_count > self.min_heap.peek().unwrap().count) {
                if self.min_heap.len() == self.k {
                    self.min_heap.pop();
                }
                self.min_heap.push(Node {
                    count: max_count,
                    item,
                });
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

        //let topk: TopK<u8> = TopK::new(k, width, depth, decay);
        let topk: TopK <Vec<u8>> = TopK::new(k, width, depth, decay);
        assert_eq!(topk.width, 100);
        assert_eq!(topk.depth, 5);
        assert_eq!(topk.decay, 0.9);
        assert_eq!(topk.buckets.len(), 5);
        assert_eq!(topk.buckets[0].len(), 100);
        assert_eq!(topk.min_heap.len(), 0);
    }

    #[test]
    fn test_query() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);
        topk.min_heap.push(Node {
            count: 1,
            item: "hello".as_bytes().to_vec(),
        });
        assert!(topk.query(&Vec::from("hello".as_bytes())));
        assert!(!topk.query(&Vec::from("world".as_bytes())));
    }

    #[test]
    fn test_count() {
        let k = 10;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Add an item with count 8
        topk.min_heap.push(Node {
            count: 8,
            item: "पद्मं".as_bytes().to_vec(),
        });
        assert_eq!(topk.count(&"पद्मं".as_bytes().to_vec()), Some(8));
        assert_eq!(topk.count(&"पुष्पं अस्ति।".as_bytes().to_vec()), None);

        // Push another item with count 1337
        topk.min_heap.push(Node {
            count: 1337,
            item: "ballynamoney".as_bytes().to_vec(),
        });
        assert_eq!(topk.count(&"ballynamoney".as_bytes().to_vec()), Some(1337));
    }

    #[test]
    fn test_add_single_item() {
        let k = 1;
        let width = 100;
        let depth = 5;
        let decay = 0.9;
        let mut topk = TopK::new(k, width, depth, decay);

        topk.add("hello".as_bytes());

        let nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();

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

        assert_eq!(topk.min_heap.len(), k); // Assertion for min_heap length

        let mut nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.item);

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

        let mut nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.item);

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

        let mut nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.item);

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

        let nodes = topk.min_heap.iter().cloned().collect::<Vec<_>>();

        assert_eq!(nodes.len(), 0);
    }

    #[test]
    fn test_add_varied_input() {
        let k = 10; // We want to track the top-10 items
        let width = 1000;
        let depth = 10;
        let decay = 0.9;

        let mut topk: TopK<Vec<u8>> = TopK::new(k, width, depth, decay);

        // Generate 100 unique items with varied addition frequencies
        let mut items_with_frequencies = Vec::new();
        for i in 0..20 {
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
        assert_eq!(topk.min_heap.len(), k, "Min-heap does not contain the top-k items");

        // Verify the min-heap contains the correct top-k items based on frequency
        // The top-k items should be the last k items added due to their higher frequencies
        let mut top_items = topk.min_heap.iter().map(|node| std::str::from_utf8(&node.item).unwrap().to_string()).collect::<Vec<_>>();
        top_items.sort(); // Sorting to simplify validation

        //let expected_top_items = (90..100)
        let expected_top_items = (10..20)
            .map(|i| format!("item{}", i))
            .collect::<Vec<_>>();

        println!("Expected top items: {:?}", expected_top_items);
        println!("Actual top items: {:?}", top_items);

        for expected_item in expected_top_items.iter() {
            assert!(top_items.contains(expected_item), "Expected item {} to be in the top-k items", expected_item);
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
        let count = topk.count(&item).unwrap_or(0);

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
        let count_item1 = topk.count(&item1).unwrap_or(0);
        let count_item2 = topk.count(&item2).unwrap_or(0);

        // Assert that both items' counts are as expected
        assert_eq!(count_item1, num_additions_item1, "The count for item1 does not match the expected value.");
        assert_eq!(count_item2, num_additions_item2, "The count for item2 does not match the expected value.");

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
        let count_frequent = topk.count(&frequent_item).unwrap_or(0);
        let count_less_frequent = topk.count(&less_frequent_item).unwrap_or(0);

        assert!(count_frequent > count_less_frequent, "Decay does not seem to have been applied correctly.");
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
        // This implies checking the buckets directly, which may require making the test part of the TopK module
        // or adding a method to TopK for testing purposes that can check the bucket states.
        assert!(topk.buckets.iter().any(|row| row.iter().any(|bucket| bucket.fingerprint == topk.hash(item) && bucket.count == 1)),
                "The item was not inserted into an empty bucket correctly.");
    }

}
