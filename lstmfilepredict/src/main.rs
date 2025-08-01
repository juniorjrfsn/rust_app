// projeto : lstmfilepredict
// file : src/main.rs - Corrigido para compatibilidade exata com lstmfiletrain
use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls};
use postgres_types::Json;
use rusqlite::{Connection, params};
use chrono::{NaiveDate, Duration};
use std::error::Error;

// Assuming LSTMError is defined or use a generic Box<dyn Error>
// For simplicity here, using Box<dyn Error> directly. You can define LSTMError if preferred.
// #[derive(Debug)] enum LSTMError { ParseError(String), ... }

#[derive(Parser, Debug)]
#[command(author, version = "2.2.0", about = "LSTM Stock Price Prediction - Enhanced Version", long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long, default_value = "../dados")]
    data_dir: String,
    #[arg(long, default_value_t = 5)]
    num_predictions: usize,
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

// =================================================================================================
// Data Structures - MUST MATCH lstmfiletrain/src/main.rs EXACTLY
// =================================================================================================

#[derive(Debug, Deserialize, Serialize)] // Add Serialize if needed elsewhere
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
    data_mean: f32,
    data_std: f32,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
    timestamp: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockData {
    date: String,
    closing: f32,
}

// =================================================================================================
// LSTM Cell Definition (MUST MATCH lstmfiletrain EXACTLY)
// =================================================================================================

#[derive(Debug)]
struct LSTMCell {
    // hidden_size: usize, // Not strictly needed if we always use shape(), but can keep if preferred
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
    // Constructor to create LSTMCell from the LSTMLayerWeights struct
    fn from_weights(weights: LSTMLayerWeights) -> Self {
        Self {
            // hidden_size: weights.w_input.shape()[0], // Optional if not used elsewhere
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

    // MUST MATCH EXACTLY the forward pass logic in lstmfiletrain
    fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        // Gate calculations including bias
        // Using `mapv` with clamping for numerical stability, matching lstmfiletrain
        let i_t = sigmoid(&(&self.w_input.dot(input) + &self.u_input.dot(h_prev) + &self.b_input));
        let f_t = sigmoid(&(&self.w_forget.dot(input) + &self.u_forget.dot(h_prev) + &self.b_forget));
        let o_t = sigmoid(&(&self.w_output.dot(input) + &self.u_output.dot(h_prev) + &self.b_output));
        let g_t = tanh(&(&self.w_cell.dot(input) + &self.u_cell.dot(h_prev) + &self.b_cell));

        // Cell state update
        let c_t = &f_t * c_prev + &i_t * &g_t;
        // Hidden state update
        let h_t = &o_t * &tanh(&c_t);

        (h_t, c_t)
    }
}

// Activation functions - MUST MATCH lstmfiletrain
fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|val| 1.0 / (1.0 + (-val.max(-50.0).min(50.0)).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|val| val.max(-50.0).min(50.0).tanh())
}

// =================================================================================================
// Data Loading Functions
// =================================================================================================

fn load_data_from_sqlite(data_dir: &str, source: &str, asset: &str) -> Result<Vec<StockData>, Box<dyn Error>> {
    let db_path = format!("{}/{}.db", data_dir, source);
    let conn = Connection::open(&db_path)?;
    let mut stmt = conn.prepare("SELECT date, closing FROM stock_records WHERE asset = ?1 ORDER BY date ASC")?;
    let rows = stmt.query_map(params![asset], |row| {
        Ok(StockData {
            date: row.get(0)?,
            closing: row.get(1)?,
        })
    })?;

    let data: Result<Vec<_>, _> = rows.collect();
    Ok(data?)
}

// =================================================================================================
// Prediction Logic
// =================================================================================================

fn predict_prices(
    model_weights: ModelWeights,
    data: Vec<StockData>,
    num_predictions: usize,
    verbose: bool,
) -> Result<Vec<(String, f32)>, Box<dyn Error>> { // Changed error type for simplicity
    let seq_length = model_weights.seq_length;
    if data.len() < seq_length {
        return Err(format!("Not enough data points. Need at least {}, got {}.", seq_length, data.len()).into());
    }

    let prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    // Normalize the last sequence using the model's stored mean and std
    let last_prices = &prices[prices.len() - seq_length..];
    let normalized_last_prices: Vec<f32> = last_prices.iter().map(|&p| (p - model_weights.data_mean) / model_weights.data_std).collect();

    // Reconstruct LSTM layers from weights
    let lstm_layers: Vec<LSTMCell> = model_weights.layers.into_iter().map(LSTMCell::from_weights).collect();

    let mut predictions = Vec::new();
    let mut current_sequence = normalized_last_prices.clone();
    
    // --- Robust Date Parsing ---
    let last_date_str = &data.last().unwrap().date;
    // Try multiple common formats
    let last_date = NaiveDate::parse_from_str(last_date_str, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%d.%m.%Y"))
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%m/%d/%Y"))
        .map_err(|e| format!("Failed to parse last date '{}' from data using known formats: {}", last_date_str, e))?;

    for i in 0..num_predictions {
        // Initialize hidden and cell states for all layers for this prediction sequence
        let mut h_states = vec![Array1::zeros(model_weights.hidden_size); lstm_layers.len()];
        let mut c_states = vec![Array1::zeros(model_weights.hidden_size); lstm_layers.len()];

        // Process the current sequence through the LSTM layers
        // Iterate through each timestep in the current_sequence
        for &input_val in &current_sequence {
            let mut layer_input = Array1::from_vec(vec![input_val]); // Input for the first layer at this timestep

            // Pass the input through each layer sequentially for this timestep
            for (layer_idx, layer) in lstm_layers.iter().enumerate() {
                // --- CORRECT MULTI-LAYER FORWARD PASS ---
                let (h_new, c_new) = layer.forward(&layer_input, &h_states[layer_idx], &c_states[layer_idx]);
                // Update states for *this* layer for the *next* timestep
                h_states[layer_idx] = h_new; // No clone needed here as h_new is not reused
                c_states[layer_idx] = c_new;
                // The output of this layer becomes the input to the *next* layer for *this* timestep
                layer_input = h_new;
            }
            // After processing all layers for this timestep, `layer_input` holds the output of the last layer.
            // This is implicitly carried forward as the input for the next timestep via the loop.
            // The final states `h_states` and `c_states` are updated and will be used for the next input_val.
        }

        // --- FINAL PREDICTION ---
        // Use the final hidden state of the *last* layer after processing the entire sequence
        let final_hidden_state = &h_states[lstm_layers.len() - 1]; // Access last layer's final h_state
        let prediction_normalized = final_hidden_state.dot(&model_weights.w_final) + model_weights.b_final; // Include bias
        let prediction_price = prediction_normalized * model_weights.data_std + model_weights.data_mean;

        let prediction_date = last_date + Duration::days(i as i64 + 1);
        let prediction_date_str = prediction_date.format("%Y-%m-%d").to_string();
        predictions.push((prediction_date_str.clone(), prediction_price));

        if verbose {
            println!("📅 {}: R$ {:.2}", prediction_date_str, prediction_price);
        }

        // --- UPDATE SEQUENCE FOR NEXT PREDICTION (Sliding Window) ---
        current_sequence.remove(0);
        current_sequence.push(prediction_normalized); // Add the new prediction (normalized)
    }

    Ok(predictions)
}

// =================================================================================================
// Main Function
// =================================================================================================

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== LSTM Stock Price Prediction (Enhanced) ===");
    println!("Asset: {}, Source: {}", args.asset, args.source);
    println!("Number of Predictions: {}", args.num_predictions);
    println!();

    // --- Load Model from PostgreSQL ---
    println!("📡 Loading model from PostgreSQL...");
    // Ensure the connection string matches your PostgreSQL setup
    let mut pg_client = Client::connect("postgres://postgres:postgres@localhost:5432/lstm_db", NoTls)?;

    // Query the specific model weights
    let query = "SELECT weights_json FROM lstm_weights_v3 WHERE asset = $1 AND source = $2";
    let rows = pg_client.query(query, &[&args.asset, &args.source])?;

    if rows.is_empty() {
        return Err(format!("No model found for asset '{}' and source '{}'. Did you train the model first?", args.asset, args.source).into());
    }

    // Deserialize the JSONB column
    let weights_json: Json<ModelWeights> = rows[0].get("weights_json");
    let model_weights: ModelWeights = weights_json.0;

    println!("✅ Model loaded successfully (Trained on: {})", model_weights.timestamp);
    println!();

    // --- Load Data from SQLite ---
    println!("📂 Loading recent data from SQLite...");
    let data = load_data_from_sqlite(&args.data_dir, &args.source, &args.asset)?;
    if data.is_empty() {
         return Err("No data loaded from SQLite.".into());
    }
    println!("✅ Loaded {} data points", data.len());
    println!();

    // --- Generate Predictions ---
    println!("🔮 Generating predictions...");
    let predictions = predict_prices(model_weights, data, args.num_predictions, args.verbose)?;
    println!("✅ Predictions generated");
    println!();

    // --- Display Final Results ---
    if !args.verbose {
        println!("--- Predictions ---");
        for (date, price) in &predictions {
            println!("📅 {}: R$ {:.2}", date, price);
        }
    }
    println!();
    println!("🎉 Prediction process completed successfully!");

    Ok(())
}

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
