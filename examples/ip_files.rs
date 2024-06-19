
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::time::Instant;
use heavykeeper::TopK;

const KEY_SIZE: usize = 13;

fn read_in_trace(trace_prefix: &str, max_item_num: usize) -> io::Result<(Vec<Vec<u8>>, HashMap<Vec<u8>, u32>)> {
    let mut count = 0;
    let mut keys = Vec::new();
    let mut actual_flow_sizes = HashMap::new();

    let datafile_cnt = 0;
    //for datafile_cnt in 0..=10 {
        let trace_file_path = format!("{}{}.dat", trace_prefix, datafile_cnt);
        println!("Start reading {}", trace_file_path);

        let file = File::open(&trace_file_path)?;
        let mut reader = BufReader::new(file);
        let mut temp = vec![0; KEY_SIZE];

        while reader.read_exact(&mut temp).is_ok() {
            let key = temp.clone();
            keys.push(key.clone());
            let counter = actual_flow_sizes.entry(key).or_insert(0);
            *counter += 1;
            count += 1;

            if count >= max_item_num {
                panic!("The dataset has more than {} items, set a larger value for max_item_num", max_item_num);
            }
        }

        println!("Finished reading {} ({} items), the dataset now has {} items", trace_file_path, count, keys.len());
    //}

    Ok((keys, actual_flow_sizes))
}

fn read_in_traces(trace_prefix: &str, max_item_num: usize) -> io::Result<(Vec<Vec<u8>>, HashMap<Vec<u8>, u32>)> {
    let mut count = 0;
    let mut keys = Vec::new();
    let mut actual_flow_sizes = HashMap::new();

    for datafile_cnt in 0..=10 {
        let trace_file_path = format!("{}{}.dat", trace_prefix, datafile_cnt);
        println!("Start reading {}", trace_file_path);

        let file = File::open(&trace_file_path)?;
        let mut reader = BufReader::new(file);
        let mut temp = vec![0; KEY_SIZE];

        while reader.read_exact(&mut temp).is_ok() {
            let key = temp.clone();
            keys.push(key.clone());
            let counter = actual_flow_sizes.entry(key).or_insert(0);
            *counter += 1;
            count += 1;

            if count >= max_item_num {
                panic!("The dataset has more than {} items, set a larger value for max_item_num", max_item_num);
            }
        }

        println!("Finish reading {} ({} items), the dataset now has {} items", trace_file_path, count, keys.len());
    }

    Ok((keys, actual_flow_sizes))
}

fn main() -> io::Result<()> {
    let max_item_num = 40 * 1_000_000;
    let (keys, actual_flow_sizes) = read_in_traces("data/", max_item_num)?;

    println!("number of items: {}", keys.len());
    println!("number of flows: {}", actual_flow_sizes.len());

    // create a topk struct
    let mut topk = TopK::new(1000, 21124, 2, 0.95);

    // add all the keys to the topk struct
    let start = Instant::now();
    for key in &keys {
        topk.add(key);
    }
    let duration = start.elapsed();

    // calculate throughput
    let num_of_seconds = duration.as_secs_f64();
    let throughput = (keys.len() as f64 / 1_000_000.0) / num_of_seconds;
    println!("use {} seconds", num_of_seconds);
    println!("throughput: {} Mpps, each insert operation uses {} ns", throughput, 1_000.0 / throughput);

    for node in topk.list() {
        // each node is a tuple of (item, count)
        // item is a 13 byte big-endian encoding of
        // protocol (1byte), src_ip (4bytes), dst_ip (4bytes), src_port (2bytes), dst_port (2bytes)
        // count is the number of times the item has been added
        // print the decoded item and the count
        //
        // Each 13-byte string is a network five-tuple in the format of (srcIP, dstIP, srcPort, dstPort, protocol).
        // For a 13-byte string "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c", the ASCII code should be:
        //
        // - srcIP = "\x00\x01\x02\x03"
        // - dstIP = "\x04\x05\x06\x07"
        // - srcPort = "\x08\x09"
        // - dstPort = "\x0a\x0b"
        // - protocol = "\x0c".

        let src_ip = format!("{}.{}.{}.{}", node.item[0], node.item[1], node.item[2], node.item[3]);
        let dst_ip = format!("{}.{}.{}.{}", node.item[4], node.item[5], node.item[6], node.item[7]);
        let src_port = u16::from_be_bytes([node.item[8], node.item[9]]);
        let dst_port = u16::from_be_bytes([node.item[10], node.item[11]]);
        let protocol = node.item[12];

        println!("{} {}:{} -> {}:{} {}", protocol, src_ip, src_port, dst_ip, dst_port, node.count);
    }

    Ok(())
}

