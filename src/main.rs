use std::io::{self, BufRead};
use std::process::exit;
use clap::Parser;
use heavykeeper::TopK;

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

    let mut topk = TopK::<String>::new(args.k, args.width, args.depth, args.decay);

    let mut process_line = |line: &str| {
        for word in line.split_whitespace() {
            topk.add(word.to_string());
        }
    };

    if args.input.is_none() {
        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut buffer = String::new();
        while stdin_lock.read_line(&mut buffer).unwrap() > 0 {
            process_line(&buffer);
            buffer.clear();
        }
    } else {
        let file = std::fs::File::open(args.input.unwrap()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            exit(1);
        });
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            process_line(&line);
        }
    }

    for node in topk.list() {
        println!("{} {}", node.item, node.count);
    }
}
