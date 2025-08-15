// projeto : lstmfilextract
// file : src/export_to_json.rs

use clap::Parser;
use env_logger;
use log::{error, info, warn};
use postgres::{Client, NoTls, Row};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
enum ExportError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] postgres::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Directory creation failed: {path}")]
    DirectoryError { path: String },
}

#[derive(Parser)]
#[command(name = "export_to_json")]
#[command(about = "Export PostgreSQL stock_records data to JSON files")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(
        long,
        help = "Database URL (e.g., postgres://postgres:postgres@localhost:5432/lstm_db)"
    )]
    db_url: String,
    #[arg(
        long,
        default_value = "../../dados/consolidado",
        help = "Output directory path"
    )]
    output_dir: String,
    #[arg(long, help = "Export specific asset (optional)")]
    asset: Option<String>,
    #[arg(
        long,
        help = "Export all assets in separate files",
        default_value = "false"
    )]
    separate_files: bool,
    #[arg(long, help = "Pretty print JSON", default_value = "true")]
    pretty: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StockRecord {
    asset: String,
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AssetData {
    asset: String,
    total_records: usize,
    date_range: Option<(String, String)>,
    records: Vec<StockRecord>,
    export_timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConsolidatedData {
    total_assets: usize,
    total_records: usize,
    assets: Vec<String>,
    export_timestamp: String,
    data: HashMap<String, Vec<StockRecord>>,
}

fn row_to_stock_record(row: &Row) -> StockRecord {
    StockRecord {
        asset: row.get("asset"),
        date: row.get("date"),
        closing: row.get("closing"),
        opening: row.get("opening"),
        high: row.get("high"),
        low: row.get("low"),
        volume: row.get("volume"),
        variation: row.get("variation"),
        created_at: row
            .get::<_, Option<std::time::SystemTime>>("created_at")
            .map(|st| {
                // Convert SystemTime to DateTime<Utc>
                let datetime: chrono::DateTime<chrono::Utc> = st.into();
                datetime.to_rfc3339()
            }),
    }
}

fn get_all_assets(client: &mut Client) -> Result<Vec<String>, ExportError> {
    info!("Fetching list of all assets...");
    let rows = client.query(
        "SELECT DISTINCT asset FROM stock_records ORDER BY asset",
        &[],
    )?;
    let assets: Vec<String> = rows.iter().map(|row| row.get("asset")).collect();
    info!("Found {} unique assets", assets.len());
    Ok(assets)
}

fn get_records_for_asset(
    client: &mut Client,
    asset: &str,
) -> Result<Vec<StockRecord>, ExportError> {
    info!("Fetching records for asset: {}", asset);
    let query = "
        SELECT asset, date, closing, opening, high, low, volume, variation, created_at
        FROM stock_records
        WHERE asset = $1
        ORDER BY date ASC
    ";
    let rows = client.query(query, &[&asset])?;
    let records: Vec<StockRecord> = rows.iter().map(|row| row_to_stock_record(row)).collect();
    info!("Found {} records for asset {}", records.len(), asset);
    Ok(records)
}

fn get_all_records(client: &mut Client) -> Result<Vec<StockRecord>, ExportError> {
    info!("Fetching all records...");
    let query = "
        SELECT asset, date, closing, opening, high, low, volume, variation, created_at
        FROM stock_records
        ORDER BY asset ASC, date ASC
    ";
    let rows = client.query(query, &[])?;
    let records: Vec<StockRecord> = rows.iter().map(|row| row_to_stock_record(row)).collect();
    info!("Found {} total records", records.len());
    Ok(records)
}

fn create_asset_data(asset: &str, records: Vec<StockRecord>) -> AssetData {
    let date_range = if !records.is_empty() {
        Some((
            records.first().unwrap().date.clone(),
            records.last().unwrap().date.clone(),
        ))
    } else {
        None
    };

    AssetData {
        asset: asset.to_string(),
        total_records: records.len(),
        date_range,
        records,
        export_timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

fn save_json_file<T: Serialize>(
    data: &T,
    file_path: &str,
    pretty: bool,
) -> Result<(), ExportError> {
    let json_data = if pretty {
        serde_json::to_string_pretty(data)?
    } else {
        serde_json::to_string(data)?
    };

    std::fs::write(file_path, json_data)?;
    info!("✅ Data successfully saved to JSON: {}", file_path);
    Ok(())
}

fn export_single_asset(cli: &Cli, client: &mut Client, asset: &str) -> Result<(), ExportError> {
    info!("Exporting data for asset: {}", asset);

    let records = get_records_for_asset(client, asset)?;
    let asset_data = create_asset_data(asset, records);

    // Create output directory if it doesn't exist
    let output_path = Path::new(&cli.output_dir);
    if !output_path.exists() {
        create_dir_all(output_path).map_err(|_| ExportError::DirectoryError {
            path: cli.output_dir.clone(),
        })?;
        info!("Created output directory: {}", cli.output_dir);
    }

    let file_path = format!("{}/{}_data.json", cli.output_dir, asset);
    save_json_file(&asset_data, &file_path, cli.pretty)?;

    println!(
        "✅ Exported {} records for asset {} to {}",
        asset_data.total_records, asset, file_path
    );
    Ok(())
}

fn export_all_assets_separate(cli: &Cli, client: &mut Client) -> Result<(), ExportError> {
    info!("Exporting all assets to separate files...");

    let assets = get_all_assets(client)?;

    // Create output directory if it doesn't exist
    let output_path = Path::new(&cli.output_dir);
    if !output_path.exists() {
        create_dir_all(output_path).map_err(|_| ExportError::DirectoryError {
            path: cli.output_dir.clone(),
        })?;
        info!("Created output directory: {}", cli.output_dir);
    }

    let mut total_records = 0;

    for asset in &assets {
        let records = get_records_for_asset(client, asset)?;
        let asset_data = create_asset_data(asset, records);
        total_records += asset_data.total_records;

        let file_path = format!("{}/{}_data.json", cli.output_dir, asset);
        save_json_file(&asset_data, &file_path, cli.pretty)?;

        println!(
            "✅ Exported {} records for asset {} to {}",
            asset_data.total_records, asset, file_path
        );
    }

    println!(
        "🎉 Export complete! {} assets, {} total records exported to separate files",
        assets.len(),
        total_records
    );
    Ok(())
}

fn export_all_assets_consolidated(cli: &Cli, client: &mut Client) -> Result<(), ExportError> {
    info!("Exporting all assets to consolidated file...");

    let records = get_all_records(client)?;
    let mut data_by_asset: HashMap<String, Vec<StockRecord>> = HashMap::new();

    // Group records by asset
    for record in records {
        data_by_asset
            .entry(record.asset.clone())
            .or_insert_with(Vec::new)
            .push(record);
    }

    // Sort records within each asset by date
    for records in data_by_asset.values_mut() {
        records.sort_by(|a, b| a.date.cmp(&b.date));
    }

    let assets: Vec<String> = data_by_asset.keys().cloned().collect();
    let total_records: usize = data_by_asset.values().map(|v| v.len()).sum();

    let consolidated_data = ConsolidatedData {
        total_assets: assets.len(),
        total_records,
        assets: {
            let mut sorted_assets = assets.clone();
            sorted_assets.sort();
            sorted_assets
        },
        export_timestamp: chrono::Utc::now().to_rfc3339(),
        data: data_by_asset,
    };

    // Create output directory if it doesn't exist
    let output_path = Path::new(&cli.output_dir);
    if !output_path.exists() {
        create_dir_all(output_path).map_err(|_| ExportError::DirectoryError {
            path: cli.output_dir.clone(),
        })?;
        info!("Created output directory: {}", cli.output_dir);
    }

    let file_path = format!("{}/consolidated_stock_data.json", cli.output_dir);
    save_json_file(&consolidated_data, &file_path, cli.pretty)?;

    println!(
        "🎉 Export complete! {} assets, {} total records exported to {}",
        consolidated_data.total_assets, consolidated_data.total_records, file_path
    );
    Ok(())
}

fn export_command(cli: Cli) -> Result<(), ExportError> {
    info!("Starting data export to JSON format");
    info!("Database URL: {}", cli.db_url);
    info!("Output directory: {}", cli.output_dir);

    // Connect to database
    let mut client = Client::connect(&cli.db_url, NoTls).map_err(|e| {
        error!("Failed to connect to database: {}", e);
        e
    })?;

    info!("✅ Successfully connected to database");

    // Verify table exists and get count
    let count_result = client.query_one("SELECT COUNT(*) as total FROM stock_records", &[])?;
    let total_count: i64 = count_result.get("total");
    info!("Found {} total records in stock_records table", total_count);

    if total_count == 0 {
        warn!("⚠️  No records found in stock_records table");
        return Ok(());
    }

    // Execute export based on CLI arguments
    match cli.asset.as_ref() {
        Some(asset) => export_single_asset(&cli, &mut client, asset)?,
        None => {
            if cli.separate_files {
                export_all_assets_separate(&cli, &mut client)?;
            } else {
                export_all_assets_consolidated(&cli, &mut client)?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();

    match export_command(cli) {
        Ok(()) => {
            info!("Export completed successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}

// Exemplos de uso:
// cargo build --release

// # Exportar tudo consolidado
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db

// # Arquivos separados por ativo
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --separate-files

// # Ativo específico
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --asset PETR4

// # Exportar todos os ativos em um arquivo consolidado
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db

// # Exportar cada ativo em arquivo separado
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --separate-files

// # Exportar ativo específico
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --asset PETR4

// # Usar diretório customizado
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --output-dir ./meus_jsons

// # JSON compacto (sem pretty print)
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/lstm_db --pretty false

// # Exemplo completo com todas as opções
// cargo run --bin export_to_json -- --db-url postgres://postgres:postgres@localhost:5432/rnn_db --output-dir ../../dados/consolidado --separate-files --pretty true
