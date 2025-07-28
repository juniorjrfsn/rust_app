
use clap::Parser;
use std::fs::{File, OpenOptions};
use std::path::Path;
use csv::ReaderBuilder;
use serde::Serialize;
use toml;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
}

#[derive(Debug, Serialize)]
struct StockRecord {
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
}

#[derive(Serialize)]
struct StockData {
    records: Vec<StockRecord>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input_file_path = format!("../dados/{}/{}.csv", args.source, args.asset);
    let output_file_path = format!("../dados/{}_{}_output.toml", args.asset, args.source);

    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        println!("Input file not found: {}", input_file_path);
        return Ok(());
    }

    let file = File::open(&input_file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.len() == 7 {
            let date = record[0].to_string();
            let closing = record[1].replace(',', ".").parse::<f32>()?;
            let opening = record[2].replace(',', ".").parse::<f32>()?;
            let high = record[3].replace(',', ".").parse::<f32>()?;
            let low = record[4].replace(',', ".").parse::<f32>()?;
            let volume = record[5].replace(',', ".").trim_end_matches('M').parse::<f32>()?.mul_add(1e6, 0.0);
            let variation = record[6].replace(',', ".").trim_end_matches('%').parse::<f32>()? / 100.0;
            records.push(StockRecord { date, closing, opening, high, low, volume, variation });
        } else {
            println!("Skipping invalid record: {:?}", record);
        }
    }

    let stock_data = StockData { records };
    let toml_data = toml::to_string_pretty(&stock_data)?;
    let output_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_file_path)?;
    std::io::Write::write_all(&mut std::io::BufWriter::new(output_file), toml_data.as_bytes())?;

    println!("Data saved to: {}", output_file_path);

    Ok(())
}


// cargo run -- --asset WEGE3 --source investing



// cd lstmfilextract

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
