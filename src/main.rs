use std::io::{self, BufRead};
use memmap2::Mmap;
use std::process::exit;
use clap::Parser;
use heavykeeper::TopK;
use memchr::memchr;

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

    if args.input.is_none() {
        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut buffer = Vec::with_capacity(1024 * 1024);
        
        while stdin_lock.read_until(b'\n', &mut buffer).unwrap() > 0 {
            process_buffer(&buffer, &mut topk);
            buffer.clear();
        }
    } else {
        let file = std::fs::File::open(args.input.unwrap()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            exit(1);
        });

        let mmap = unsafe { Mmap::map(&file) }.unwrap_or_else(|e| {
            eprintln!("Error mapping file: {}", e);
            exit(1);
        });

        process_mmap(&mmap, &mut topk);
    }

    for node in topk.list() {
        println!("{} {}", node.item, node.count);
    }
}

fn process_buffer(buffer: &[u8], topk: &mut TopK<String>) {
    let mut pos = 0;
    let len = buffer.len();
    
    let mut words = Vec::with_capacity(1024*1024);

    while pos < len {
        while pos < len && (buffer[pos] == b' ' || buffer[pos] == b'\n') {
            pos += 1;
        }
        
        if pos == len {
            break;
        }

        let start = pos;
        while pos < len && buffer[pos] != b' ' && buffer[pos] != b'\n' {
            pos += 1;
        }

        if start < pos {
            let mut word = String::with_capacity(pos - start);
            unsafe {
                word.as_mut_vec().extend_from_slice(&buffer[start..pos]);
            }
            words.push(word);
            
            if words.len() >= 1024 {
                for word in words.drain(..) {
                    topk.add(word);
                }
            }
        }
    }

    for word in words {
        topk.add(word);
    }
}

fn process_mmap(mmap: &[u8], topk: &mut TopK<String>) {
    let mut pos = 0;
    let len = mmap.len();
    let mut words = Vec::with_capacity(1024);

    while pos < len {
        // Skip any whitespace
        while pos < len && (mmap[pos] == b' ' || mmap[pos] == b'\n') {
            pos += 1;
        }

        if pos >= len {
            break;
        }

        // Find next space using memchr
        let word_start = pos;
        pos = if let Some(space_pos) = memchr(b' ', &mmap[pos..len]) {
            word_start + space_pos
        } else {
            len
        };

        // Create word more efficiently
        let word_len = pos - word_start;
        let mut word = String::with_capacity(word_len);
        unsafe {
            let vec = word.as_mut_vec();
            vec.set_len(word_len);
            std::ptr::copy_nonoverlapping(
                mmap.as_ptr().add(word_start),
                vec.as_mut_ptr(),
                word_len
            );
        }
        words.push(word);
        
        // Batch process when buffer is full
        if words.len() >= 1024 {
            for word in words.drain(..) {
                topk.add(word);
            }
        }

        pos += 1;
    }

    // Process remaining words
    for word in words {
        topk.add(word);
    }
}
