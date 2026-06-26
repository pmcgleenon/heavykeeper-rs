/// Compares mem_bytes() estimation approaches against actual heap usage measured by dhat.
///
/// Reproduces the table from issue #75 (baseline + mem_bytes_with rows), then adds two
/// new rows for the corrected-HashMap-formula variants.
///
/// Run with:
///   cargo run --example mem_compare

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use heavykeeper::BucketedTopK;

fn make_item(idx: usize, size: usize) -> Vec<u8> {
    let mut item = vec![0u8; size];
    // Encode index into the first 4 bytes so every item is unique.
    item[0] = (idx & 0xFF) as u8;
    item[1] = ((idx >> 8) & 0xFF) as u8;
    item[2] = ((idx >> 16) & 0xFF) as u8;
    item[3] = ((idx >> 24) & 0xFF) as u8;
    item
}

struct Row {
    label: String,
    baseline: usize,
    with_items: usize,
    corrected: usize,
    corrected_with: usize,
    dhat_actual: usize,
}

fn measure(label: &str, k: usize, width: usize, depth: usize, item_size: usize) -> Row {
    let _profiler = dhat::Profiler::builder().testing().build();

    // Snapshot before any sketch allocation.
    let before = dhat::HeapStats::get();

    let mut sketch = BucketedTopK::<Vec<u8>>::new(k, width, depth, 0.9);

    // Insert each of k unique items many times so all k slots fill up.
    // 20 rounds is enough: after 20 passes each item has count ~20 and the
    // queue fills up well before that; extra passes only update existing counts.
    let rounds = 20;
    for round in 0..(k * rounds) {
        let item = make_item(round % k, item_size);
        sketch.add(&item, 1);
    }

    // Snapshot after population; live allocations = sketch heap.
    let after = dhat::HeapStats::get();
    let dhat_actual = after.curr_bytes - before.curr_bytes;

    let baseline = sketch.mem_bytes();
    let with_items = sketch.mem_bytes_with(|v| v.capacity());
    let corrected = sketch.mem_bytes_corrected();
    let corrected_with = sketch.mem_bytes_corrected_with(|v| v.capacity());

    Row {
        label: label.to_string(),
        baseline,
        with_items,
        corrected,
        corrected_with,
        dhat_actual,
    }
}

fn pct(estimate: usize, actual: usize) -> String {
    if actual == 0 {
        return "n/a".to_string();
    }
    let diff = actual as isize - estimate as isize;
    format!("{:+.1}%", diff as f64 / actual as f64 * 100.0)
}

fn print_table(rows: &[Row]) {
    let w = [38, 10, 10, 10, 15, 12];
    let sep = format!(
        "+-{}-+-{}-+-{}-+-{}-+-{}-+-{}-+",
        "-".repeat(w[0]),
        "-".repeat(w[1]),
        "-".repeat(w[2]),
        "-".repeat(w[3]),
        "-".repeat(w[4]),
        "-".repeat(w[5]),
    );

    println!("{sep}");
    println!(
        "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} |",
        "Config",
        "baseline",
        "+items",
        "+hashmap",
        "+items+hashmap",
        "dhat actual",
        w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
    );
    println!(
        "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} |",
        "",
        "mem_bytes()",
        "mem_bytes_with()",
        "mem_bytes_corrected()",
        "mem_bytes_corrected_with()",
        "",
        w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
    );
    println!("{sep}");

    for r in rows {
        println!(
            "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} |",
            r.label,
            format!("{} B", r.baseline),
            format!("{} B", r.with_items),
            format!("{} B", r.corrected),
            format!("{} B", r.corrected_with),
            format!("{} B", r.dhat_actual),
            w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
        );
        println!(
            "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {:>w5$} |",
            "",
            pct(r.baseline, r.dhat_actual),
            pct(r.with_items, r.dhat_actual),
            pct(r.corrected, r.dhat_actual),
            pct(r.corrected_with, r.dhat_actual),
            "(actual)",
            w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
        );
        println!("{sep}");
    }
}

fn main() {
    println!("\nBucketedTopK memory estimation: approaches vs dhat actual");
    println!("T = Vec<u8>, decay = 0.9, default depth = 5\n");

    let rows = vec![
        measure("baseline  (k=1k,  w=8k,  item=12)", 1_000, 8_192, 5, 12),
        measure("long items (k=1k,  w=8k,  item=32)", 1_000, 8_192, 5, 32),
        measure("long items (k=1k,  w=8k,  item=64)", 1_000, 8_192, 5, 64),
        measure("large k    (k=10k, w=8k,  item=12)", 10_000, 8_192, 5, 12),
        measure("large k    (k=50k, w=8k,  item=12)", 50_000, 8_192, 5, 12),
        measure("small width (k=1k,  w=1k,  item=12)", 1_000, 1_024, 5, 12),
        measure("large width (k=1k,  w=64k, item=12)", 1_000, 65_536, 5, 12),
    ];

    print_table(&rows);

    println!("\nColumn key:");
    println!("  baseline             = current mem_bytes() from PR #85");
    println!("  +items               = mem_bytes_with(|v| v.capacity())  [issue proposal]");
    println!("  +hashmap             = mem_bytes_corrected()              [fixed raw_cap formula]");
    println!("  +items+hashmap       = mem_bytes_corrected_with(...)      [both fixes combined]");
}
