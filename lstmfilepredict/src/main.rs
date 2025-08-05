// projeto: lstmfilepredict
// file: src/main.rs



mod rna {
    pub mod model;
    pub mod data;
    pub mod utils;
}

use clap::Parser;
use log::{info, error, warn};
use env_logger;
use postgres::{Client, NoTls, Row};
use rna::data::data::{load_data_from_postgres, predict_prices};
use rna::utils::utils::mask_password;

#[derive(thiserror::Error, Debug)]
pub enum LSTMError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] postgres::Error),
    #[error("Model not found for asset '{0}'")]
    ModelNotFound(String),
    #[error("Insufficient data for asset '{asset}': required {required}, actual {actual}")]
    InsufficientData { asset: String, required: usize, actual: usize },
    #[error("Data loading error: {0}")]
    DataLoadingError(String),
    #[error("Date parsing error: {0}")]
    DateParseError(String),
    #[error("Prediction error: {0}")]
    PredictionError(String),
    #[error("Model forward pass error: {0}")]
    ForwardError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

#[derive(Parser, Debug)]
#[command(author, version = "3.0.2", about = "LSTM Stock Price Prediction - Detailed Version compatible with lstmfiletrain", long_about = None)]
struct Args {
    #[arg(long, help = "Asset to predict (e.g., WEGE). If not provided, predicts for all assets.")]
    asset: Option<String>,
    #[arg(long, default_value_t = 5, help = "Number of future predictions (5 to 7 days)")]
    num_predictions: usize,
    #[arg(long, default_value_t = false, help = "Enable verbose logging")]
    verbose: bool,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(if cfg!(debug_assertions) { log::LevelFilter::Debug } else { log::LevelFilter::Info })
        .init();

    let args = Args::parse();
    println!("🚀 Starting LSTM Stock Price Prediction (Detailed Log)");
    println!("==================================================");

    match &args.asset {
        Some(asset_name) => info!("🎯 Target Asset: {}", asset_name),
        None => info!("🎯 Target: ALL assets in the database"),
    }
    info!("🔢 Number of Predictions Requested: {}", args.num_predictions);
    info!("🔊 Verbose Mode: {}", args.verbose);
    info!("🔗 PostgreSQL Connection String: {}", mask_password(&args.pg_conn));

    info!("📡 Connecting to PostgreSQL database...");
    let mut pg_client = Client::connect(&args.pg_conn, NoTls)?;
    info!("✅ Successfully connected to PostgreSQL");

    let models_to_process = if let Some(asset_prefix) = &args.asset {
        info!("📥 Loading model for specific asset '{}'...", asset_prefix);
        vec![(load_model_by_asset_prefix(&mut pg_client, asset_prefix)?)]
    } else {
        info!("📥 Loading models for ALL assets...");
        load_all_models(&mut pg_client)?
    };

    if models_to_process.is_empty() {
        if args.asset.is_some() {
            error!("❌ No model found for the specified asset prefix '{}'.", args.asset.as_ref().unwrap());
            println!("\n❌ Prediction process failed: Model not found for '{}'", args.asset.as_ref().unwrap());
        } else {
            println!("\n⚠️ Prediction process completed: No models found in the database.");
        }
        return Ok(());
    }

    info!("📊 Starting prediction loop for {} asset(s)...", models_to_process.len());
    println!("\n🔮 Starting Predictions");
    println!("======================");

    let mut successful_predictions = 0;
    let mut failed_predictions = 0;

    for (full_asset_name, model_weights) in models_to_process {
        println!("\n--- 📊 Processing Asset: {} ---", full_asset_name);
        info!("📈 Beginning prediction process for asset: {}", full_asset_name);

        info!("📂 Loading recent historical data for {} from PostgreSQL...", full_asset_name);
        let data_result = load_data_from_postgres(&mut pg_client, &full_asset_name, model_weights.seq_length, args.verbose);
        let data = match data_result {
            Ok(d) => d,
            Err(e) => {
                error!("❌ Failed to load data for {}: {}", full_asset_name, e);
                failed_predictions += 1;
                continue;
            }
        };

        if data.is_empty() {
            warn!("⚠️ No data loaded for {}. Skipping prediction.", full_asset_name);
            failed_predictions += 1;
            continue;
        }

        info!("🔮 Generating {} future predictions for {}...", args.num_predictions, full_asset_name);
        match predict_prices(model_weights, data, args.num_predictions, args.verbose) {
            Ok(predictions) => {
                info!("✅ Predictions successfully generated for {}", full_asset_name);
                successful_predictions += 1;

                println!("\n--- 🎯 Final Predictions for {} ---", full_asset_name);
                for (date, price) in &predictions {
                    println!("📅 {}: R$ {:.2}", date, price);
                }
                println!("--- End of predictions for {} ---", full_asset_name);
            }
            Err(e) => {
                error!("❌ Failed to generate predictions for {}: {}", full_asset_name, e);
                failed_predictions += 1;
            }
        }
    }

    println!("\n🏁 Prediction Process Summary");
    println!("==============================");
    println!("✅ Successful Assets: {}", successful_predictions);
    println!("❌ Failed Assets: {}", failed_predictions);
    println!("📊 Total Assets Processed: {}", successful_predictions + failed_predictions);
    println!("\n🎉 Prediction process completed!");
    Ok(())
}

fn load_model_by_asset_prefix(client: &mut Client, asset_prefix: &str) -> Result<(String, rna::model::model::ModelWeights), LSTMError> {
    info!("  🔍 Searching for model with asset prefix: '{}'", asset_prefix);
    let query = "SELECT asset, weights_json FROM lstm_weights_v3 WHERE asset LIKE $1 || '%' AND source = $2 ORDER BY created_at DESC LIMIT 1";
    let search_pattern = format!("{}%", asset_prefix);

    let row_opt = client.query_opt(query, &[&search_pattern, &"investing"])?;

    if let Some(row) = row_opt {
        let full_asset_name: String = row.get("asset");
        info!("  ✅ Found model record for asset: {}", full_asset_name);
        let weights_json_str: String = row.get("weights_json");
        info!("  📦 Deserializing model weights from JSON...");
        let weights: rna::model::model::ModelWeights = serde_json::from_str(&weights_json_str)
            .map_err(|e| LSTMError::DeserializationError(format!("Failed to deserialize model for {}: {}", full_asset_name, e)))?;
        info!("  ✅ Model successfully loaded and deserialized for asset: {}", full_asset_name);
        Ok((full_asset_name, weights))
    } else {
        Err(LSTMError::ModelNotFound(asset_prefix.to_string()))
    }
}

fn load_all_models(client: &mut Client) -> Result<Vec<(String, rna::model::model::ModelWeights)>, LSTMError> {
    info!("  📡 Fetching all models from lstm_weights_v3 table...");
    let query = "SELECT asset, weights_json FROM lstm_weights_v3 WHERE source = $1 ORDER BY asset";
    let rows: Vec<Row> = client.query(query, &[&"investing"])?;

    info!("  📊 Found {} model records in the database", rows.len());
    let mut models = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let full_asset_name: String = row.get("asset");
        info!("  🔧 Processing model record {}/{}: {}", i + 1, rows.len(), full_asset_name);
        let weights_json_str: String = row.get("weights_json");
        match serde_json::from_str::<rna::model::model::ModelWeights>(&weights_json_str) {
            Ok(weights) => {
                models.push((full_asset_name.clone(), weights));
                info!("  ✅ Model {} successfully loaded and deserialized", full_asset_name);
            }
            Err(e) => {
                error!("  ❌ Failed to deserialize model for asset {}: {}", full_asset_name, e);
            }
        }
    }

    if models.is_empty() && !rows.is_empty() {
        warn!("⚠️ Models found in DB but none could be successfully loaded.");
    } else if models.is_empty() {
        warn!("⚠️ No models found in the database for source 'investing'.");
    } else {
        info!("  ✅ Successfully loaded {} models", models.len());
    }

    Ok(models)
}


// Single Asset:  
// cargo run --release -- --asset WEGE --num-predictions 5 --verbose

// All Assets: 
// cargo run --release -- --num-predictions 5 --verbose
  

 // cd ~/Documentos/projetos/rust_app/lstmfilepredict


// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose


// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose
 
// cargo run --release -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 50 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005
 

// cargo run --release -- --asset "SLCE3 Dados Históricos" --num-predictions 5 --verbose


// USIM5 Dados Históricos
// ISAE4 Dados Históricos
// TAEE4 Dados Históricos
// PETR4 Dados Históricos
// EGIE3 Dados Históricos
// WEGE3 Dados Históricos
// VALE3 Dados Históricos
// SLCE3 Dados Históricos

// cargo run --release -- --asset WEGE3 --num-predictions 5 --verbose