// projeto: lstmfilepredict
// file: src/main.rs  

use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls};
use postgres_types::Json;
use chrono::{NaiveDate, Duration};
use thiserror::Error;
use log::info;
use env_logger;

#[derive(Error, Debug)]
#[allow(dead_code)]
enum LSTMError {
    #[error("Insufficient data for asset {asset}: need at least {required}, got {actual}")]
    InsufficientData { asset: String, required: usize, actual: usize },
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
    #[error("Date parsing error: {0}")]
    DateParseError(String),
    #[error("No model found for asset {0}")]
    ModelNotFound(String),
    #[error("Prediction error: {0}")]
    PredictionError(String),
}

#[derive(Parser, Debug)]
#[command(author, version = "3.0.0", about = "LSTM Stock Price Prediction - Enhanced Version compatible with lstmfiletrain", long_about = None)]
struct Args {
    #[arg(long, help = "Asset to predict (e.g., SLCE3)")]
    asset: String,
    #[arg(long, default_value = "../dados", help = "Data directory (for compatibility, not used)")]
    data_dir: String,
    #[arg(long, default_value_t = 5, help = "Number of future predictions")]
    num_predictions: usize,
    #[arg(long, default_value_t = false, help = "Enable verbose logging")]
    verbose: bool,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct StockData {
    date: String,
    closing: f32,
    opening: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct LSTMLayerWeights {
    w_input: Array2<f32>,
    u_input: Array2<f32>,
    b_input: Array1<f32>,
    w_forget: Array2<f32>,
    u_forget: Array2<f32>,
    b_forget: Array1<f32>,
    w_output: Array2<f32>,
    u_output: Array2<f32>,
    b_output: Array1<f32>,
    w_cell: Array2<f32>,
    u_cell: Array2<f32>,
    b_cell: Array1<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelWeights {
    asset: String,
    source: String,
    layers: Vec<LSTMLayerWeights>,
    w_final: Array1<f32>,
    b_final: f32,
    closing_mean: f32,
    closing_std: f32,
    opening_mean: f32,
    opening_std: f32,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
    timestamp: String,
}

#[derive(Debug)]
struct LSTMCell {
    w_input: Array2<f32>,
    u_input: Array2<f32>,
    b_input: Array1<f32>,
    w_forget: Array2<f32>,
    u_forget: Array2<f32>,
    b_forget: Array1<f32>,
    w_output: Array2<f32>,
    u_output: Array2<f32>,
    b_output: Array1<f32>,
    w_cell: Array2<f32>,
    u_cell: Array2<f32>,
    b_cell: Array1<f32>,
}

impl LSTMCell {
    fn from_weights(weights: LSTMLayerWeights) -> Self {
        Self {
            w_input: weights.w_input,
            u_input: weights.u_input,
            b_input: weights.b_input,
            w_forget: weights.w_forget,
            u_forget: weights.u_forget,
            b_forget: weights.b_forget,
            w_output: weights.w_output,
            u_output: weights.u_output,
            b_output: weights.b_output,
            w_cell: weights.w_cell,
            u_cell: weights.u_cell,
            b_cell: weights.b_cell,
        }
    }

    fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let i_t = (&self.w_input.dot(input) + &self.u_input.dot(h_prev) + &self.b_input).mapv(|x| Self::sigmoid(x));
        let f_t = (&self.w_forget.dot(input) + &self.u_forget.dot(h_prev) + &self.b_forget).mapv(|x| Self::sigmoid(x));
        let o_t = (&self.w_output.dot(input) + &self.u_output.dot(h_prev) + &self.b_output).mapv(|x| Self::sigmoid(x));
        let g_t = (&self.w_cell.dot(input) + &self.u_cell.dot(h_prev) + &self.b_cell).mapv(|x| Self::tanh(x));
        
        let c_t = &f_t * c_prev + &i_t * &g_t;
        let h_t = &o_t * &c_t.mapv(|x| Self::tanh(x));
        
        (h_t, c_t)
    }

    fn sigmoid(x: f32) -> f32 {
        if x > 500.0 { return 1.0; }
        if x < -500.0 { return 0.0; }
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f32) -> f32 {
        if x > 20.0 { return 1.0; }
        if x < -20.0 { return -1.0; }
        x.tanh()
    }
}

fn normalize_data(prices: &[f32], mean: f32, std: f32) -> Vec<f32> {
    prices.iter().map(|x| (x - mean) / std.max(1e-8)).collect()
}

fn load_data_from_postgres(pg_client: &mut Client, asset: &str) -> Result<Vec<StockData>, LSTMError> {
    let rows = pg_client.query(
        "SELECT date, closing, opening 
         FROM stock_records 
         WHERE asset = $1 
         ORDER BY date ASC",
        &[&asset],
    )?;
    
    let records: Vec<StockData> = rows.into_iter().map(|row| StockData {
        date: row.get(0),
        closing: row.get(1),
        opening: row.get(2),
    }).collect();
    
    Ok(records)
}

fn create_sequence(data: &[StockData], seq_length: usize, closing_mean: f32, closing_std: f32, opening_mean: f32, opening_std: f32) -> Result<Vec<f32>, LSTMError> {
    if data.len() < seq_length {
        return Err(LSTMError::InsufficientData {
            asset: data.get(0).map_or("unknown".to_string(), |d| d.date.clone()),
            required: seq_length,
            actual: data.len(),
        });
    }

    let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
    
    let norm_closing = normalize_data(&closing_prices, closing_mean, closing_std);
    let norm_opening = normalize_data(&opening_prices, opening_mean, opening_std);
    
    let mut sequence = Vec::new();
    for i in 0..seq_length {
        sequence.push(norm_closing[i]);
        sequence.push(norm_opening[i]);
        if i >= 4 {
            let ma5 = closing_prices[(i-4)..=i].iter().sum::<f32>() / 5.0;
            sequence.push((ma5 - closing_mean) / closing_std.max(1e-8));
        } else {
            sequence.push(norm_closing[i]);
        }
        sequence.push(if i > 0 { (closing_prices[i] - closing_prices[i-1]) / closing_std.max(1e-8) } else { 0.0 });
    }
    
    Ok(sequence)
}

fn predict_prices(
    model_weights: ModelWeights,
    data: Vec<StockData>,
    num_predictions: usize,
    verbose: bool,
) -> Result<Vec<(String, f32)>, LSTMError> {
    let seq_length = model_weights.seq_length;
    if data.len() < seq_length {
        return Err(LSTMError::InsufficientData {
            asset: model_weights.asset,
            required: seq_length,
            actual: data.len(),
        });
    }

    let lstm_layers: Vec<LSTMCell> = model_weights.layers.into_iter().map(LSTMCell::from_weights).collect();
    let mut predictions = Vec::new();
    let last_date_str = &data.last().unwrap().date;
    let last_date = NaiveDate::parse_from_str(last_date_str, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%d.%m.%Y"))
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%m/%d/%Y"))
        .map_err(|e| LSTMError::DateParseError(format!("Failed to parse date '{}': {}", last_date_str, e)))?;

    let mut current_sequence_data = data[data.len() - seq_length..].to_vec();
    
    for i in 0..num_predictions {
        let current_sequence = create_sequence(
            &current_sequence_data,
            seq_length,
            model_weights.closing_mean,
            model_weights.closing_std,
            model_weights.opening_mean,
            model_weights.opening_std,
        )?;
        
        if verbose {
            info!("Prediction {}: Created sequence with {} timesteps (4 features each)", i + 1, seq_length);
        }

        let mut h_states = vec![Array1::zeros(model_weights.hidden_size); model_weights.num_layers];
        let mut c_states = vec![Array1::zeros(model_weights.hidden_size); model_weights.num_layers];

        for j in (0..current_sequence.len()).step_by(4) {
            let layer_input = Array1::from_vec(vec![
                current_sequence[j],     // Normalized closing
                current_sequence[j + 1], // Normalized opening
                current_sequence[j + 2], // Normalized moving average
                current_sequence[j + 3], // Normalized momentum
            ]);

            let mut next_input = layer_input.clone();
            for (layer_idx, layer) in lstm_layers.iter().enumerate() {
                let (h_new, c_new) = layer.forward(&next_input, &h_states[layer_idx], &c_states[layer_idx]);
                h_states[layer_idx] = h_new.clone();
                c_states[layer_idx] = c_new;
                next_input = h_new;
            }
        }

        let final_hidden_state = &h_states[lstm_layers.len() - 1];
        let prediction_normalized = final_hidden_state.dot(&model_weights.w_final) + model_weights.b_final;
        let prediction_price = prediction_normalized * model_weights.closing_std + model_weights.closing_mean;

        let prediction_date = last_date + Duration::days(i as i64 + 1);
        let prediction_date_str = prediction_date.format("%Y-%m-%d").to_string();
        predictions.push((prediction_date_str.clone(), prediction_price));

        if verbose {
            info!("📅 {}: Predicted R$ {:.2} (Normalized: {:.4})", prediction_date_str, prediction_price, prediction_normalized);
        }

        // Update sequence for next prediction
        let new_data_point = StockData {
            date: prediction_date_str,
            closing: prediction_price,
            opening: prediction_price, // Use predicted closing as opening for simplicity
        };
        current_sequence_data.remove(0);
        current_sequence_data.push(new_data_point);
    }

    Ok(predictions)
}

fn main() -> Result<(), LSTMError> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    info!("Starting LSTM Stock Price Prediction (Enhanced)");
    info!("Asset: {}, Number of Predictions: {}, Verbose: {}", args.asset, args.num_predictions, args.verbose);

    // Load model from PostgreSQL
    info!("📡 Loading model from PostgreSQL...");
    let mut pg_client = Client::connect(&args.pg_conn, NoTls)?;
    let query = "SELECT weights_json FROM lstm_weights_v3 WHERE asset = $1 AND source = $2";
    let rows = pg_client.query(query, &[&args.asset, &"investing"])?;
    
    if rows.is_empty() {
        return Err(LSTMError::ModelNotFound(format!("No model found for asset '{}'", args.asset)));
    }

    let weights_json: Json<ModelWeights> = rows[0].get("weights_json");
    let model_weights: ModelWeights = weights_json.0;
    info!("✅ Model loaded successfully (Trained on: {})", model_weights.timestamp);

    // Load data from PostgreSQL
    info!("📂 Loading recent data from PostgreSQL...");
    let data = load_data_from_postgres(&mut pg_client, &args.asset)?;
    if data.is_empty() {
        return Err(LSTMError::PredictionError("No data loaded from PostgreSQL.".to_string()));
    }
    info!("✅ Loaded {} data points", data.len());

    // Generate predictions
    info!("🔮 Generating predictions...");
    let predictions = predict_prices(model_weights, data, args.num_predictions, args.verbose)?;
    info!("✅ Predictions generated");

    // Display results
    if !args.verbose {
        println!("--- Predictions for {} ---", args.asset);
        for (date, price) in &predictions {
            println!("📅 {}: R$ {:.2}", date, price);
        }
    }
    println!("🎉 Prediction process completed successfully!");

    Ok(())
}




// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose


// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose
 
// cargo run --release -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 50 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005
 

// cargo run --release -- --asset "SLCE3 Dados Históricos" --num-predictions 5 --verbose

 

// cargo run -- --asset WEGE3 --source investing --num-predictions 5 --verbose
 
// Exemplo de uso:
// cargo run -- --asset WEGE3 --source investing --num-predictions 5 --verbose
// cargo run -- --asset WEGE3 --source investing --num-predictions 10


// cd lstmfilepredict
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --num-predictions 20

// cargo run -- predict --asset WEGE3 --source investing --num-predictions 20 
// cargo run -- --asset WEGE3 --source investing

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
