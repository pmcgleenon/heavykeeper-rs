# heavykeeper-rs

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
[![Build Status][actions-badge]][actions-url]

[crates-badge]: https://img.shields.io/crates/v/heavykeeper.svg
[crates-url]: https://crates.io/crates/heavykeeper
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/pmcgleenon/heavykeeper-rs/blob/master/LICENSE
[actions-badge]: https://github.com/pmcgleenon/heavykeeper-rs/actions/workflows/rust.yml/badge.svg
[actions-url]: https://github.com/pmcgleenon/heavykeeper-rs/actions?query=workflow%3Arust+branch%3Amain

Top-K Heavykeeper algorithm for Top-K elephant flows

This is based on the [paper](https://www.usenix.org/system/files/conference/atc18/atc18-gong.pdf)
HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant Flows
by Junzhi Gong, Tong Yang, Haowei Zhang, and Hao Li, Peking University;
Steve Uhlig, Queen Mary, University of London; Shigang Chen, University of Florida;
Lorna Uden, Staffordshire University; Xiaoming Li, Peking University

# Example

See [ip_files.rs](examples/ip_files.rs) for an example of how to use the library to 
count the top-k IP flows in a file.

A sample usage is as follows:
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

# Other Implementations

| Name                       | Language | Github Repo                                                                  |
|----------------------------|----------|------------------------------------------------------------------------------|
| SegmentIO                  | Go       | https://github.com/segmentio/topk                                            |
| Aegis                      | Go       | https://github.com/go-kratos/aegis/blob/main/topk/heavykeeper.go             |
| Tomasz Kolaj               | Go       | https://github.com/migotom/heavykeeper                                       |
| HeavyKeeper Paper          | C++      | https://github.com/papergitkeeper/heavy-keeper-project                       |
| Jigsaw-Sketch              | C++      | https://github.com/duyang92/jigsaw-sketch-paper/tree/main/CPU/HeavyKeeper    |
| Redis Bloom Heavykeeper    | C        | https://github.com/RedisBloom/RedisBloom/blob/master/src/topk.c              |
| Count-Min-Sketch           | Rust     | https://github.com/alecmocatta/streaming_algorithms                          |
| Sliding Window HeavyKeeper | Go       | https://github.com/keilerkonzept/topk                                        |

# Running

An example driver program which can be used as a word count program can be found at [`main.rs`](src/main.rs).

Usage:
```
cargo build --release
target/release/heavykeeper -k 10 -w 8192 -d 2 -y 0.95 -f data/war_and_peace.txt
```

## Building the example 
```
cargo build --examples --release
target/release/examples/ip_files
```

# License
This project is dual licensed under the Apache/MIT license.   
