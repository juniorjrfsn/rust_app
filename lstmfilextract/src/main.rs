// projeto : lstmfilextract
// file : src/main.rs
 

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
use postgres::{Client, NoTls};
use std::fs;

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
    DatabaseError(#[from] postgres::Error),
}

#[derive(Parser)]
#[command(name = "lstm_extract")]
#[command(about = "Extract and convert CSV data to TOML format and PostgreSQL database")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(long, help = "Data source (e.g., investing)")]
    source: String,
    #[arg(long, default_value = "../dados", help = "Data directory path")]
    data_dir: String,
    #[arg(long, help = "Skip TOML output generation")]
    skip_toml: bool,
    #[arg(long, help = "Database URL (e.g., postgres://postgres:postgres@localhost:5432/lstm_db)")]
    db_url: String,
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
        .flexible(true)
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




fn save_to_database(records: &[StockRecord], asset: &str, client: &mut Client) -> Result<(), LSTMError> {
    // Create table if it doesn't exist
    client.batch_execute(
        "CREATE TABLE IF NOT EXISTS stock_records (
            asset TEXT NOT NULL,
            date TEXT NOT NULL,
            closing REAL NOT NULL,
            opening REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            volume REAL NOT NULL,
            variation REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset, date)
        )",
    )?;

    // Insert or update records
    for record in records {
        client.execute(
            "INSERT INTO stock_records (asset, date, closing, opening, high, low, volume, variation)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT ON CONSTRAINT stock_records_pkey
             DO UPDATE SET
                 closing = EXCLUDED.closing,
                 opening = EXCLUDED.opening,
                 high = EXCLUDED.high,
                 low = EXCLUDED.low,
                 volume = EXCLUDED.volume,
                 variation = EXCLUDED.variation,
                 created_at = CURRENT_TIMESTAMP",
            &[
                &asset,
                &record.date,
                &record.closing,
                &record.opening,
                &record.high,
                &record.low,
                &record.volume,
                &record.variation,
            ],
        )?;
    }
    
    info!("Data successfully saved to PostgreSQL for asset: {}", asset);
    Ok(())
}




fn extract_command(cli: Cli) -> Result<(), LSTMError> {
    info!("Starting data extraction from {}", cli.source);
    
    let data_dir = Path::new(&cli.data_dir);
    if !data_dir.exists() || !data_dir.is_dir() {
        return Err(LSTMError::FileNotFound { path: cli.data_dir.clone() });
    }

    let mut all_records = Vec::new();
    let mut total_skipped = 0;
    let mut total_errors = 0;

    // List all CSV files in the directory
    let entries = fs::read_dir(&cli.data_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().and_then(|s| s.to_str()) == Some("csv")
        })
        .collect::<Vec<_>>();

    if entries.is_empty() {
        return Err(LSTMError::InvalidCsv { msg: "No CSV files found in directory".into() });
    }

    let mut client = Client::connect(&cli.db_url, NoTls)?;

    for entry in entries {
        let file_path = entry.path();
        let file_name = file_path.file_stem().unwrap().to_str().unwrap();
        let asset = file_name.split('_').next().unwrap_or(file_name).to_string();

        let parsed_data = parse_csv_data(file_path.to_str().unwrap(), &cli.date_format)?;
        let mut records = parsed_data.valid_records;

        sort_records_by_date(&mut records, &cli.date_format);
        
        let date_range = if !records.is_empty() {
            Some((records.first().unwrap().date.clone(), records.last().unwrap().date.clone()))
        } else {
            None
        };

        let stock_data = StockData {
            asset: asset.clone(),
            source: cli.source.clone(),
            total_records: records.len(),
            date_range,
            records: records.clone(),
        };

        if !cli.skip_toml {
            let output_file_path = format!("{}/{}_{}_output.toml", cli.data_dir, asset, cli.source);
            save_to_toml(&stock_data, &output_file_path)?;
        }

        save_to_database(&records, &asset, &mut client)?;

        all_records.extend(records);
        total_skipped += parsed_data.skipped_count;
        total_errors += parsed_data.error_count;

        info!("Processed {} records for asset {}", stock_data.total_records, asset);
    }

    println!("✅ Extraction complete! {} total records processed", all_records.len());
    if !cli.skip_toml {
        println!("   📄 TOML files generated in {}", cli.data_dir);
    }
    println!("   🗄️  Database updated");
    if total_skipped > 0 || total_errors > 0 {
        println!("   ⚠️  Skipped {} records, {} errors", total_skipped, total_errors);
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



// # Uso básico
// cargo run -- --source investing --db-url postgres://postgres:postgres@localhost:5432/lstm_db
 
 


// # Processamento paralelo com configuração customizada
// cargo run -- --source investing --config config.toml --parallel --verbose

// # Validação sem salvar (dry run)
// cargo run -- --source investing --dry-run --verbose

// # Pular operações de banco
// cargo run -- --source investing --skip-db --data-dir ./meus-dados



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
