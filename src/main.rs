use std::io::{self, BufRead, stdin};
use std::process::exit;

use heavykeeper::TopK;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short = 'k')]
    k: usize,

    #[arg(short = 'w', default_value_t = 8)]
    width: usize,

    #[arg(short = 'd', default_value_t = 2048)]
    depth: usize,

    #[arg(short = 'y', default_value_t = 0.9)]
    decay: f64,

    #[arg(short = 'f')]
    input: Option<String>,

}
fn main() {

    let args = Args::parse();

    let mut topk = TopK::new(args.k, args.width, args.depth, args.decay);

    if args.input.is_none() {
        let stdin = stdin();
        for line in stdin.lock().lines() {
            let item = line.unwrap();
            // break item into words
            for word in item.split_whitespace() {
                topk.add(word.as_bytes());
            }
        }
    } else {
        let file = std::fs::File::open(args.input.unwrap());
        if file.is_err() {
            eprintln!("Error: {}", file.err().unwrap());
            exit(1);
        }
        let file = file.unwrap();
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            let item = line.unwrap();
            // break item into words
            for word in item.split_whitespace() {
                topk.add(word.as_bytes());
            }
        }
    }

    
    for node in topk.list() {
        println!("{} {}", String::from_utf8_lossy(&node.item), node.count);
    }
}
