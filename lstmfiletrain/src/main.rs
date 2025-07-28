


use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use std::path::Path;
use ndarray::{Array1, Array2, s};
use log::{info, error};
use env_logger;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockRecord {
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
}

// Simplified LSTM Cell
#[derive(Serialize)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Vec<Vec<f32>>, // Input to hidden weights (serialized as Vec<Vec<f32>> for JSON)
    weight_hh: Vec<Vec<f32>>, // Hidden to hidden weights
    bias: Vec<f32>,           // Bias terms
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> LSTMCell {
        let mut rng = rand::thread_rng();
        let weight_ih = (0..4 * hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let weight_hh = (0..4 * hidden_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let bias = (0..4 * hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect();
        LSTMCell { input_size, hidden_size, weight_ih, weight_hh, bias }
    }

    fn forward(&self, x: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let gates = {
            let ih = Array2::from_shape_vec((4 * self.hidden_size, self.input_size), self.weight_ih.iter().flatten().cloned().collect()).unwrap();
            let hh = Array2::from_shape_vec((4 * self.hidden_size, self.hidden_size), self.weight_hh.iter().flatten().cloned().collect()).unwrap();
            let bias = Array1::from_vec(self.bias.clone());
            &ih.dot(x) + &hh.dot(h_prev) + &bias
        };
        let i = sigmoid(&gates.slice(s![..self.hidden_size]).to_owned());
        let f = sigmoid(&gates.slice(s![self.hidden_size..2 * self.hidden_size]).to_owned());
        let o = sigmoid(&gates.slice(s![2 * self.hidden_size..3 * self.hidden_size]).to_owned());
        let g = tanh(&gates.slice(s![3 * self.hidden_size..]).to_owned());

        let c = &f * c_prev + &i * &g;
        let h = &o * tanh(&c);

        (h.to_owned(), c.to_owned())
    }
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let input_file_path = format!("../dados/{}_{}_output.json", args.asset, args.source);
    let output_model_path = format!("../dados/{}_{}_lstm_model.json", args.asset, args.source);

    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        error!("Input file not found: {}", input_file_path);
        return Ok(());
    }

    info!("Loading JSON data from {}", input_file_path);
    let json_data = fs::read_to_string(&input_file_path)?;
    let records: Vec<StockRecord> = serde_json::from_str(&json_data)?;

    // Extract closing prices
    let prices: Array1<f32> = Array1::from_vec(records.iter().map(|r| r.closing).collect());
    info!("Loaded {} price records", prices.len());

    // Normalize data
    let mean = prices.mean().unwrap_or(0.0);
    let std = prices.std_axis(ndarray::Axis(0), 0.0).into_scalar();
    let normalized_prices = prices.mapv(|x| (x - mean) / std);

    // Prepare sequences
    let seq_length = 10;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    for i in 0..(normalized_prices.len() - seq_length) {
        let seq = normalized_prices.slice(s![i..i + seq_length]).to_vec();
        let target = normalized_prices[i + seq_length];
        sequences.push(seq);
        targets.push(target);
    }

    let hidden_size = 10;
    let mut lstm = LSTMCell::new(1, hidden_size);

    // Initial hidden and cell states
    let mut h_t = Array1::zeros(hidden_size);
    let mut c_t = Array1::zeros(hidden_size);

    // Training loop (simple gradient descent)
    let learning_rate = 0.01;
    for epoch in 0..50 {
        let mut total_loss = 0.0;
        for i in 0..sequences.len() {
            // Reset hidden and cell states for each sequence
            h_t = Array1::zeros(hidden_size);
            c_t = Array1::zeros(hidden_size);
            
            // Process the full sequence
            for &input_val in &sequences[i] {
                let x = Array1::from_vec(vec![input_val]);
                let (h_next, c_next) = lstm.forward(&x, &h_t, &c_t);
                h_t = h_next;
                c_t = c_next;
            }
            
            let prediction = h_t[0]; // Use final hidden state as prediction
            let target = targets[i];
            let loss = (prediction - target).powi(2);
            total_loss += loss;
        }
        info!("Epoch: {}, Loss: {:.4}", epoch, total_loss / sequences.len() as f32);
    }

    // Save model weights as JSON
    let model_state = serde_json::to_value(&lstm)?;
    fs::write(&output_model_path, serde_json::to_string_pretty(&model_state)?)?;
    info!("Model saved to: {}", output_model_path);

    Ok(())
}

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
// cargo run -- --asset WEGE3 --source investing



// cd lstmfiletrain