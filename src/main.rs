use std::env;
use std::io::{self, BufRead};

use heavykeeper::TopK;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <k> <width> <depth> <decay>", args[0]);
        std::process::exit(1);
    }

    let k: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid value for k: {}", args[1]);
        std::process::exit(1);
    });

    let width: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Invalid value for width: {}", args[2]);
        std::process::exit(1);
    });

    let depth: usize = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Invalid value for depth: {}", args[3]);
        std::process::exit(1);
    });

    let decay: f64 = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Invalid value for decay: {}", args[4]);
        std::process::exit(1);
    });


    let mut topk = TopK::new(k, width, depth, decay);

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let item = line.unwrap();
        // break item into words
        for word in item.split_whitespace() {
            topk.add(word.as_bytes());
        }
    }

    for node in topk.list() {
        println!("{} {}", String::from_utf8_lossy(&node.item), node.count);
    }
}
