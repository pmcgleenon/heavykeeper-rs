use std::io::{self, BufRead};
use memmap2::Mmap;
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

    if args.input.is_none() {
        let stdin = io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut buffer = Vec::with_capacity(1024 * 1024);
        
        while stdin_lock.read_until(b'\n', &mut buffer).unwrap() > 0 {
            process_bytes(&buffer, &mut topk);
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

        process_bytes(&mmap, &mut topk);
    }

    for node in topk.list() {
        println!("{} {}", node.item, node.count);
    }
}

/// Scans a byte slice for contiguous alphabetic words, converts them to lowercase, and updates the provided TopK structure.
///
/// The function iterates over the input bytes, skipping any non-alphabetic characters. When it encounters a sequence of
/// alphabetic characters, it treats them as a word, converts the word to lowercase, and adds it to the TopK instance. A
/// single, reusable string buffer is employed to minimize memory allocations during processing.
///
/// # Examples
///
/// ```
/// use my_crate::{process_bytes, TopK};
///
/// let data = b"Hello, Rust! Learn and explore Rust programming.";
/// let mut topk = TopK::new(10, 64, 0.5);
/// process_bytes(data, &mut topk);
///
/// // Assuming `TopK` tracks added words, it should now contain the following lowercase words.
/// assert!(topk.contains("hello"));
/// assert!(topk.contains("rust"));
/// assert!(topk.contains("learn"));
/// assert!(topk.contains("and"));
/// assert!(topk.contains("explore"));
/// assert!(topk.contains("programming"));
/// ```
fn process_bytes(bytes: &[u8], topk: &mut TopK<String>) {
    let mut pos = 0;
    let len = bytes.len();
    let mut word = String::with_capacity(64);  // Single reusable String

    while pos < len {
        // Skip non-alphabetic characters
        while pos < len && !bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }

        if pos >= len {
            break;
        }

        // Find end of word
        let word_start = pos;
        while pos < len && bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }

        let word_len = pos - word_start;
        if word_len > 0 {
            // Clear and reuse the string
            word.clear();
            
            // Reserve space if needed
            if word.capacity() < word_len {
                word.reserve(word_len - word.capacity());
            }

            // SAFETY: we know we're only dealing with ASCII alphabetic characters
            unsafe {
                word.as_mut_vec().extend(bytes[word_start..pos].iter().map(|&b| b.to_ascii_lowercase()));
                topk.add(word.clone());
            }
        }
    }
}
