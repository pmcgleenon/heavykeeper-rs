use heavykeeper::TopK;

fn main() {
    // Create a new TopK with:
    // - k=10 (number of top items to track)
    // - width=1000 (size of hash table, larger values use more memory but reduce collisions)
    // - depth=4 (number of hash functions, more depth increases accuracy but uses more CPU)
    // - decay=0.9 (decay factor for frequency counting, higher values give more weight to recent items)

    let mut topk: TopK<Vec<u8>> = TopK::new(10, 1000, 4, 0.9);

    // Add some example items multiple times to show frequency counting
    for _ in 0..5 {
        topk.add(&b"frequent item".to_vec());
    }

    for _ in 0..3 {
        topk.add(&b"less frequent item".to_vec());
    }

    topk.add(&b"rare item".to_vec());

    // Print the items and their counts in order of frequency
    println!("Top items and their frequencies:");
    for node in topk.list() {
        println!("{}: {}", String::from_utf8_lossy(&node.item), node.count);
    }

    // Demonstrate the count() method
    let item = b"frequent item".to_vec();
    println!(
        "\nCount for '{}': {}",
        String::from_utf8_lossy(&item),
        topk.count(&item)
    );

    // Demonstrate the query() method
    println!(
        "Is '{}' in top-k? {}",
        String::from_utf8_lossy(&item),
        if topk.query(&item) { "yes" } else { "no" }
    );
}
