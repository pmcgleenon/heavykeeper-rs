use heavykeeper::TopK;

fn main() {
    // Create a new TopK with k=10, width=1000, depth=4, decay=0.9
    let mut topk: TopK<Vec<u8>> = TopK::new(10, 1000, 4, 0.9);

    // Add some example items multiple times to show frequency counting
    for _ in 0..5 {
        topk.add(b"frequent item".to_vec());
    }
    
    for _ in 0..3 {
        topk.add(b"less frequent item".to_vec());
    }
    
    topk.add(b"rare item".to_vec());

    // Print the items and their counts in order of frequency
    println!("Top items and their frequencies:");
    for node in topk.list() {
        println!("{}: {}", String::from_utf8_lossy(&node.item), node.count);
    }

    // Demonstrate the count() method
    let item = b"frequent item".to_vec();
    println!("\nCount for '{}': {}", String::from_utf8_lossy(&item), topk.count(&item));

    // Demonstrate the query() method
    println!("Is '{}' in top-k? {}", 
        String::from_utf8_lossy(&item),
        if topk.query(&item) { "yes" } else { "no" });
}
