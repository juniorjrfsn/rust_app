// projeto : lstmfilextract
// file : src/main.rs

use clap::{Parser};
use std::fs::{File, OpenOptions};
use std::path::Path;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use toml;
use log::{info, warn, error};
use env_logger;
use thiserror::Error;
use chrono::NaiveDate;
use rusqlite::{Connection, params, Transaction};

#[derive(Error, Debug)]
enum LSTMError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    #[error("Invalid CSV format: {msg}")]
    InvalidCsv { msg: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Date parse error: {0}")]
    DateParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

#[derive(Parser)]
#[command(name = "lstm_extract")]
#[command(about = "Extract and convert CSV data to TOML format and SQLite database")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(long, help = "Stock symbol (e.g., WEGE3)")]
    asset: String,
    #[arg(long, help = "Data source (e.g., investing)")]
    source: String,
    #[arg(long, default_value = "../dados", help = "Data directory path")]
    data_dir: String,
    #[arg(long, help = "Skip TOML output generation")]
    skip_toml: bool,
    #[arg(long, help = "Skip SQLite database storage")]
    skip_db: bool,
    #[arg(long, help = "Date format pattern", default_value = "%d.%m.%Y")]
    date_format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StockRecord {
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct StockData {
    asset: String,
    source: String,
    total_records: usize,
    date_range: Option<(String, String)>,
    records: Vec<StockRecord>,
}

#[derive(Debug)]
struct ParsedData {
    valid_records: Vec<StockRecord>,
    skipped_count: usize,
    error_count: usize,
}

fn parse_float(s: &str) -> Result<f32, LSTMError> {
    let cleaned = s.replace(',', ".").replace(" ", "").trim().to_string();
    if cleaned.is_empty() {
        return Err(LSTMError::ParseError("Empty value".into()));
    }
    cleaned.parse::<f32>()
        .map_err(|_| LSTMError::ParseError(format!("Invalid number: {}", s)))
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let binding = s.replace(',', ".").replace(" ", "");
    let cleaned = binding.trim();
    
    if cleaned.is_empty() {
        return Ok(0.0); // Allow empty volume
    }
    
    let multiplier = if cleaned.ends_with('M') || cleaned.ends_with('m') {
        1e6
    } else if cleaned.ends_with('K') || cleaned.ends_with('k') {
        1e3
    } else if cleaned.ends_with('B') || cleaned.ends_with('b') {
        1e9
    } else {
        1.0
    };
    
    let number_part = cleaned.trim_end_matches(|c: char| c.is_alphabetic());
    let value = number_part.parse::<f32>()
        .map_err(|_| LSTMError::ParseError(format!("Invalid volume: {}", s)))?;
    Ok(value * multiplier)
}

fn parse_percentage(s: &str) -> Result<f32, LSTMError> {
    let binding = s.replace(',', ".").replace(" ", "");
    let cleaned = binding.trim_end_matches('%');
    
    if cleaned.is_empty() {
        return Ok(0.0); // Allow empty variation
    }
    
    let value = cleaned.parse::<f32>()
        .map_err(|_| LSTMError::ParseError(format!("Invalid percentage: {}", s)))?;
    Ok(value / 100.0)
}

fn parse_date(date_str: &str, format: &str) -> Result<NaiveDate, LSTMError> {
    NaiveDate::parse_from_str(date_str, format)
        .map_err(|e| LSTMError::DateParseError(format!("Failed to parse date '{}' with format '{}': {}", date_str, format, e)))
}

fn validate_record(record: &StockRecord) -> Result<(), LSTMError> {
    if record.closing <= 0.0 || record.opening <= 0.0 || record.high <= 0.0 || record.low <= 0.0 {
        return Err(LSTMError::ParseError("Prices must be positive".into()));
    }
    
    if record.high < record.low {
        return Err(LSTMError::ParseError("High price cannot be less than low price".into()));
    }
    
    if record.closing > record.high || record.closing < record.low {
        return Err(LSTMError::ParseError("Closing price must be between high and low".into()));
    }
    
    if record.opening > record.high || record.opening < record.low {
        return Err(LSTMError::ParseError("Opening price must be between high and low".into()));
    }
    
    Ok(())
}

fn parse_csv_data(file_path: &str, date_format: &str) -> Result<ParsedData, LSTMError> {
    info!("Reading CSV file: {}", file_path);
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .flexible(true) // Allow variable number of columns
        .from_reader(file);
    
    let mut valid_records = Vec::new();
    let mut skipped_count = 0;
    let mut error_count = 0;
    
    for (line_num, result) in rdr.records().enumerate() {
        match result {
            Ok(record) => {
                if record.len() >= 7 {
                    match parse_single_record(&record, date_format) {
                        Ok(stock_record) => {
                            if let Err(e) = validate_record(&stock_record) {
                                warn!("Validation failed for record at line {}: {}", line_num + 1, e);
                                skipped_count += 1;
                            } else {
                                valid_records.push(stock_record);
                            }
                        }
                        Err(e) => {
                            warn!("Skipping invalid record at line {}: {}", line_num + 1, e);
                            skipped_count += 1;
                        }
                    }
                } else {
                    warn!("Skipping record with insufficient columns at line {}: expected 7, got {}", line_num + 1, record.len());
                    skipped_count += 1;
                }
            }
            Err(e) => {
                error!("Error reading record at line {}: {}", line_num + 1, e);
                error_count += 1;
            }
        }
    }
    
    Ok(ParsedData {
        valid_records,
        skipped_count,
        error_count,
    })
}

fn parse_single_record(record: &csv::StringRecord, date_format: &str) -> Result<StockRecord, LSTMError> {
    let date = record[0].to_string();
    
    // Validate date format
    parse_date(&date, date_format)?;
    
    let closing = parse_float(&record[1])?;
    let opening = parse_float(&record[2])?;
    let high = parse_float(&record[3])?;
    let low = parse_float(&record[4])?;
    let volume = parse_volume(&record[5])?;
    let variation = parse_percentage(&record[6])?;
    
    Ok(StockRecord { 
        date, 
        closing, 
        opening, 
        high, 
        low, 
        volume, 
        variation 
    })
}

fn sort_records_by_date(records: &mut Vec<StockRecord>, date_format: &str) {
    records.sort_by(|a, b| {
        let date_a = parse_date(&a.date, date_format)
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
        let date_b = parse_date(&b.date, date_format)
            .unwrap_or_else(|_| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
        date_a.cmp(&date_b)
    });
}

fn save_to_toml(stock_data: &StockData, output_path: &str) -> Result<(), LSTMError> {
    let toml_data = toml::to_string_pretty(stock_data)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;
    
    let output_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)?;
    
    std::io::Write::write_all(&mut std::io::BufWriter::new(output_file), toml_data.as_bytes())?;
    info!("Data successfully saved to TOML: {}", output_path);
    Ok(())
}

fn save_to_database(records: &[StockRecord], asset: &str, db_path: &str) -> Result<(), LSTMError> {
    let conn = Connection::open(db_path)?;
    
    // Create table with indexes for better performance
    conn.execute(
        "CREATE TABLE IF NOT EXISTS stock_records (
            asset TEXT,
            date TEXT,
            closing REAL,
            opening REAL,
            high REAL,
            low REAL,
            volume REAL,
            variation REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset, date)
        )",
        [],
    )?;
    
    // Create indexes for better query performance
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_asset_date ON stock_records(asset, date)",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_date ON stock_records(date)",
        [],
    )?;
    
    // Use transaction for better performance and consistency
    let tx = conn.unchecked_transaction()?;
    
    // Clear existing data for this asset
    tx.execute("DELETE FROM stock_records WHERE asset = ?1", params![asset])?;
    
    // Scope the statement to ensure it's dropped before commit
    {
        let mut stmt = tx.prepare(
            "INSERT INTO stock_records (asset, date, closing, opening, high, low, volume, variation) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
        )?;
        
        for record in records {
            stmt.execute(params![
                asset,
                &record.date,
                record.closing,
                record.opening,
                record.high,
                record.low,
                record.volume,
                record.variation
            ])?;
        }
    } // stmt is dropped here
    
    tx.commit()?;
    info!("Data successfully saved to SQLite database: {}", db_path);
    Ok(())
}

fn extract_command(cli: Cli) -> Result<(), LSTMError> {
    info!("Starting data extraction for {} from {}", cli.asset, cli.source);
    
    let input_file_path = format!("{}/{}/{}.csv", cli.data_dir, cli.source, cli.asset);
    let output_file_path = format!("{}/{}_{}_output.toml", cli.data_dir, cli.asset, cli.source);
    let db_file_path = format!("{}/{}.db", cli.data_dir, cli.source);
    
    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        return Err(LSTMError::FileNotFound { path: input_file_path });
    }
    
    // Parse CSV data
    let parsed_data = parse_csv_data(&input_file_path, &cli.date_format)?;
    
    if parsed_data.valid_records.is_empty() {
        return Err(LSTMError::InvalidCsv { 
            msg: format!("No valid records found. Skipped: {}, Errors: {}", 
                        parsed_data.skipped_count, parsed_data.error_count)
        });
    }
    
    let mut records = parsed_data.valid_records;
    
    // Sort records by date
    sort_records_by_date(&mut records, &cli.date_format);
    
    // Create date range info
    let date_range = if !records.is_empty() {
        Some((records.first().unwrap().date.clone(), records.last().unwrap().date.clone()))
    } else {
        None
    };
    
    info!("Successfully parsed {} valid records (skipped: {}, errors: {})", 
          records.len(), parsed_data.skipped_count, parsed_data.error_count);
    
    let stock_data = StockData { 
        asset: cli.asset.clone(),
        source: cli.source.clone(),
        total_records: records.len(),
        date_range,
        records: records.clone()
    };
    
    // Save to TOML if not skipped
    if !cli.skip_toml {
        save_to_toml(&stock_data, &output_file_path)?;
    }
    
    // Save to database if not skipped
    if !cli.skip_db {
        save_to_database(&records, &cli.asset, &db_file_path)?;
    }
    
    println!("✅ Extraction complete! {} records processed", records.len());
    if !cli.skip_toml {
        println!("   📄 TOML: {}", output_file_path);
    }
    if !cli.skip_db {
        println!("   🗄️  Database: {}", db_file_path);
    }
    if parsed_data.skipped_count > 0 || parsed_data.error_count > 0 {
        println!("   ⚠️  Skipped {} records, {} errors", parsed_data.skipped_count, parsed_data.error_count);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    let cli = Cli::parse();
    
    match extract_command(cli) {
        Ok(()) => {
            info!("Command completed successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}

// Example usage:
// cargo run -- --asset WEGE3 --source investing
// cargo run -- --asset WEGE3 --source investing --skip-toml
// cargo run -- --asset WEGE3 --source investing --date-format "%Y-%m-%d"

// Example usage:
// cargo run -- --asset WEGE3 --source investing
// cargo run -- --asset WEGE3 --source investing --skip-toml
// cargo run -- --asset WEGE3 --source investing --date-format "%Y-%m-%d"

// cargo run -- --asset WEGE3 --source investing

 

 // cd lstmfilextract

// # 1. Extract data
// cargo run -- extract --asset WEGE3 --source investing

// # 2. Train the LSTM model
// cargo run -- train --asset WEGE3 --source investing --seq-length 20 --hidden-size 50

// # 3. Generate predictions
// cargo run -- predict --asset WEGE3 --source investing --num-predictions 20


// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
