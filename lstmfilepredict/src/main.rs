// projeto : lstmfilepredict
// file : src/main.rs


use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls};
use postgres_types::Json;
use rusqlite::{Connection, params};
use chrono::{NaiveDate, Duration};
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version, about = "LSTM Stock Price Prediction", long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long, default_value_t = 20)]
    seq_length: usize,
    #[arg(long, default_value_t = 20)]
    num_predictions: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockData {
    date: String,
    closing: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelWeights {
    asset: String,
    source: String,
    w_input: Array2<f32>,
    b_input: Array1<f32>,
    w_forget: Array2<f32>,
    b_forget: Array1<f32>,
    w_output: Array2<f32>,
    b_output: Array1<f32>,
    w_cell: Array2<f32>,
    b_cell: Array1<f32>,
    w_final: Array1<f32>,
    b_final: f32,
    mean: f32,
    std: f32,
    seq_length: usize,
    hidden_size: usize,
    timestamp: String,
}

struct SimpleLSTM {
    hidden_size: usize,
    w_input: Array2<f32>,
    w_forget: Array2<f32>,
    w_output: Array2<f32>,
    w_cell: Array2<f32>,
    b_input: Array1<f32>,
    b_forget: Array1<f32>,
    b_output: Array1<f32>,
    b_cell: Array1<f32>,
    w_final: Array1<f32>,
    b_final: f32,
}

impl SimpleLSTM {
    fn from_weights(weights: &ModelWeights) -> Self {
        SimpleLSTM {
            hidden_size: weights.hidden_size,
            w_input: weights.w_input.clone(),
            w_forget: weights.w_forget.clone(),
            w_output: weights.w_output.clone(),
            w_cell: weights.w_cell.clone(),
            b_input: weights.b_input.clone(),
            b_forget: weights.b_forget.clone(),
            b_output: weights.b_output.clone(),
            b_cell: weights.b_cell.clone(),
            w_final: weights.w_final.clone(),
            b_final: weights.b_final,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-80.0, 80.0)).exp())
    }

    fn tanh(x: f32) -> f32 {
        x.clamp(-80.0, 80.0).tanh()
    }

    fn forward(&self, sequence: &[f32]) -> f32 {
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);

        for &input in sequence {
            let mut x_h = Array1::zeros(1 + self.hidden_size);
            x_h[0] = input;
            for i in 0..self.hidden_size {
                x_h[1 + i] = h[i];
            }

            let i_gate = (&self.w_input.dot(&x_h) + &self.b_input).mapv(Self::sigmoid);
            let f_gate = (&self.w_forget.dot(&x_h) + &self.b_forget).mapv(Self::sigmoid);
            let o_gate = (&self.w_output.dot(&x_h) + &self.b_output).mapv(Self::sigmoid);
            let g_gate = (&self.w_cell.dot(&x_h) + &self.b_cell).mapv(Self::tanh);

            c = &f_gate * &c + &i_gate * &g_gate;
            h = &o_gate * &c.mapv(Self::tanh);
        }

        self.w_final.dot(&h) + self.b_final
    }
}

fn load_data_from_sqlite(data_dir: &str, source: &str, asset: &str) -> Result<Vec<StockData>, Box<dyn Error>> {
    let db_path = format!("{}/{}.db", data_dir, source);
    let conn = Connection::open(&db_path)?;

    let mut stmt = conn.prepare(
        "SELECT date, closing FROM stock_records WHERE asset = ?1 ORDER BY date ASC",
    )?;

    let records: Result<Vec<StockData>, _> = stmt.query_map(params![asset], |row| {
        Ok(StockData {
            date: row.get(0)?,
            closing: row.get(1)?,
        })
    })?.collect();

    Ok(records?)
}

fn validate_normalization(records: &[StockData], mean: f32, std: f32) {
    let prices: Vec<f32> = records.iter().map(|r| r.closing).collect();
    let calc_mean = prices.iter().sum::<f32>() / prices.len() as f32;
    let calc_std = (prices.iter().map(|x| (x - calc_mean).powi(2)).sum::<f32>() / prices.len() as f32).sqrt();

    println!("\n=== Data Normalization Validation ===");
    println!("Metadata Mean: {:.2}, Calculated Mean: {:.2}", mean, calc_mean);
    println!("Metadata Std: {:.2}, Calculated Std: {:.2}", std, calc_std);

    if (mean - calc_mean).abs() > 0.1 || (std - calc_std).abs() > 0.1 {
        println!("⚠️ Warning: Metadata mean/std differ significantly from calculated values!");
    }
}

fn predict_future_prices(lstm: &SimpleLSTM, sequence: &[f32], num_predictions: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut predictions = Vec::new();
    let mut current_sequence = sequence.to_vec();
    let seq_length = sequence.len();

    for i in 0..num_predictions {
        let output = lstm.forward(&current_sequence);
        let denormalized = output * std + mean;
        predictions.push(denormalized);
        println!("Prediction {}: Normalized = {:.4}, Denormalized = {:.2}", i + 1, output, denormalized);

        if current_sequence.len() >= seq_length {
            current_sequence.remove(0);
        }
        current_sequence.push(output);
    }

    predictions
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== LSTM Stock Price Prediction ===");
    println!("Asset: {}, Source: {}", args.asset, args.source);
    println!("Sequence Length: {}, Number of Predictions: {}", args.seq_length, args.num_predictions);

    // Connect to PostgreSQL
    let pg_conn_string = "postgres://postgres:postgres@localhost:5432/lstm_db";
    let mut pg_client = Client::connect(pg_conn_string, NoTls)?;

    // Load model weights from PostgreSQL
    let query = "SELECT weights_json FROM lstm_weights WHERE asset = $1 AND source = $2";
    let rows = pg_client.query(query, &[&args.asset, &args.source])?;

    if rows.is_empty() {
        println!("❌ No model found for asset {} and source {}", args.asset, args.source);
        return Ok(());
    }

    let weights_json: Json<ModelWeights> = rows[0].get(0);
    let weights = weights_json.0;
    let lstm = SimpleLSTM::from_weights(&weights);

    // Load historical data from SQLite
    let data = load_data_from_sqlite("../dados", &args.source, &args.asset)?;

    // Validate normalization
    validate_normalization(&data, weights.mean, weights.std);

    // Normalize recent prices (take the most recent seq_length records, already in chronological order)
    let recent_prices: Vec<f32> = data
        .iter()
        .rev() // Get most recent first
        .take(args.seq_length)
        .rev() // Back to chronological order
        .map(|r| {
            let normalized = (r.closing - weights.mean) / weights.std;
            println!("Date: {}, Closing: {:.2}, Normalized: {:.4}", r.date, r.closing, normalized);
            normalized
        })
        .collect();

    if recent_prices.len() < args.seq_length {
        println!("❌ Insufficient data: found {} prices, need {}", recent_prices.len(), args.seq_length);
        return Ok(());
    }

    // Test prediction for a known date (e.g., 28/07/2025)
    println!("\n=== Validation Prediction ===");
    let validation_sequence: Vec<f32> = data
        .iter()
        .rev()
        .skip(1) // Skip the last day (28/07/2025)
        .take(args.seq_length)
        .rev()
        .map(|r| (r.closing - weights.mean) / weights.std)
        .collect();
    let validation_prediction = predict_future_prices(&lstm, &validation_sequence, 1, weights.mean, weights.std);
    println!("Predicted Price for 28/07/2025: {:.2} BRL (Actual: 36.73 BRL)", validation_prediction[0]);

    // Make future predictions
    println!("\n=== Price Predictions ===");
    let predictions = predict_future_prices(&lstm, &recent_prices, args.num_predictions, weights.mean, weights.std);

    // Generate dates starting from 29/07/2025
    let start_date = NaiveDate::from_ymd_opt(2025, 7, 29).unwrap();
    for (i, price) in predictions.iter().enumerate() {
        let date = start_date + Duration::days(i as i64);
        println!("  {}: Predicted Price = {:.2} BRL", date.format("%Y-%m-%d"), price);
    }

    Ok(())
}


// cd lstmfilepredict
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --num-predictions 20

// cargo run -- predict --asset WEGE3 --source investing --num-predictions 20 
// cargo run -- --asset WEGE3 --source investing

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
