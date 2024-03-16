# heavykeeper-rs
Top-K Heavykeeper algorithm for Top-K elephant flows

This is based on the [paper ](https://www.usenix.org/system/files/conference/atc18/atc18-gong.pdf)
HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant Flows
by Junzhi Gong, Tong Yang, Haowei Zhang, and Hao Li, Peking University;
Steve Uhlig, Queen Mary, University of London; Shigang Chen, University of Florida;
Lorna Uden, Staffordshire University; Xiaoming Li, Peking University

# Example

```
    // create a new TopK

    let mut topk = TopK::new(k, width, depth, decay);

    // add some items
    topk.add(item);

    // check the counts
    for node in topk.list() {
        println!("{} {}", String::from_utf8_lossy(&node.item), node.count);
    }

```


# Running

An example driver program can be found at `main.rs`

Usage:
```
target/release/heavykeeper -k 10 -d 8 -w 8192 -y 0.9 -f data/war_and_peace.txt
```

# License
This project is licensed under the MIT license.
