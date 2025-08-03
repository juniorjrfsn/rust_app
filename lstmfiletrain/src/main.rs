// projeto: lstmfiletrain
// file: src/main.rs - Enhanced LSTM for Stock Price Prediction with PostgreSQL for All Assets

use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use log::{info, error};
use env_logger;
use thiserror::Error;
use chrono::Utc;
use postgres::{Client, NoTls};
use postgres::types::Json;
use rand::{rngs::ThreadRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

#[derive(Error, Debug)]
#[allow(dead_code)]
enum LSTMError {
    #[error("Insufficient data for asset {asset}: need at least {required}, got {actual}")]
    InsufficientData { asset: String, required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
    #[error("Training error: {0}")]
    TrainingError(String),
}

#[derive(Parser)]
#[command(name = "lstm_train")]
#[command(about = "Enhanced LSTM model for stock price prediction using PostgreSQL data for all assets")]
#[command(version = "3.0.0")]
struct Cli {
    #[arg(long, default_value = "../dados", help = "Data directory (for compatibility, not used for data loading)")]
    data_dir: String,
    #[arg(long, default_value_t = 30, help = "Sequence length")]
    seq_length: usize,
    #[arg(long, default_value_t = 0.001, help = "Learning rate")]
    learning_rate: f32,
    #[arg(long, default_value_t = 100, help = "Training epochs")]
    epochs: usize,
    #[arg(long, default_value_t = 128, help = "Hidden layer size")]
    hidden_size: usize,
    #[arg(long, default_value_t = 0.8, help = "Train/validation split")]
    train_split: f32,
    #[arg(long, default_value_t = 32, help = "Batch size")]
    batch_size: usize,
    #[arg(long, default_value_t = 0.2, help = "Dropout rate")]
    dropout_rate: f32,
    #[arg(long, default_value_t = 2, help = "Number of LSTM layers")]
    num_layers: usize,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StockData {
    date: String,
    closing: f32,
    opening: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelWeights {
    asset: String,
    source: String, // Kept for compatibility, set to "investing"
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
struct TrainingMetrics {
    asset: String,
    source: String, // Kept for compatibility, set to "investing"
    final_loss: f32,
    final_val_loss: f32,
    directional_accuracy: f32,
    mape: f32,
    rmse: f32,
    epochs_trained: usize,
    training_time: f64,
    timestamp: String,
}

mod model {
    use super::{Array1, Array2, Distribution, Normal, ThreadRng, Cli, ModelWeights, LSTMLayerWeights, Utc};

    pub struct LSTMCell {
        hidden_size: usize,
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
        pub fn new(input_size: usize, hidden_size: usize, rng: &mut ThreadRng) -> Self {
            let xavier_input = (2.0 / input_size as f32).sqrt();
            let xavier_hidden = (2.0 / hidden_size as f32).sqrt();
            let normal_input = Normal::new(0.0, xavier_input).unwrap();
            let normal_hidden = Normal::new(0.0, xavier_hidden).unwrap();

            Self {
                hidden_size,
                w_input: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
                u_input: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
                b_input: Array1::zeros(hidden_size),
                w_forget: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
                u_forget: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
                b_forget: Array1::ones(hidden_size),
                w_output: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
                u_output: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
                b_output: Array1::zeros(hidden_size),
                w_cell: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
                u_cell: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
                b_cell: Array1::zeros(hidden_size),
            }
        }

        pub fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
            let i_t = (&self.w_input.dot(input) + &self.u_input.dot(h_prev) + &self.b_input)
                .mapv(Self::sigmoid);
            let f_t = (&self.w_forget.dot(input) + &self.u_forget.dot(h_prev) + &self.b_forget)
                .mapv(Self::sigmoid);
            let o_t = (&self.w_output.dot(input) + &self.u_output.dot(h_prev) + &self.b_output)
                .mapv(Self::sigmoid);
            let g_t = (&self.w_cell.dot(input) + &self.u_cell.dot(h_prev) + &self.b_cell)
                .mapv(Self::tanh);
            
            let c_t = &f_t * c_prev + &i_t * &g_t;
            let h_t = &o_t * &c_t.mapv(Self::tanh);
            
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

        pub fn to_weights(&self) -> LSTMLayerWeights {
            LSTMLayerWeights {
                w_input: self.w_input.clone(),
                u_input: self.u_input.clone(),
                b_input: self.b_input.clone(),
                w_forget: self.w_forget.clone(),
                u_forget: self.u_forget.clone(),
                b_forget: self.b_forget.clone(),
                w_output: self.w_output.clone(),
                u_output: self.u_output.clone(),
                b_output: self.b_output.clone(),
                w_cell: self.w_cell.clone(),
                u_cell: self.u_cell.clone(),
                b_cell: self.b_cell.clone(),
            }
        }
    }

    pub struct MultiLayerLSTM {
        layers: Vec<LSTMCell>,
        pub w_final: Array1<f32>,
        pub b_final: f32,
        dropout_rate: f32,
    }

    impl MultiLayerLSTM {
        pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, dropout_rate: f32, rng: &mut ThreadRng) -> Self {
            let mut layers = Vec::new();
            
            for i in 0..num_layers {
                let layer_input_size = if i == 0 { input_size } else { hidden_size };
                layers.push(LSTMCell::new(layer_input_size, hidden_size, rng));
            }

            let xavier_final = (2.0 / hidden_size as f32).sqrt();
            let normal_final = Normal::new(0.0, xavier_final).unwrap();
            let w_final = Array1::from_shape_fn(hidden_size, |_| normal_final.sample(rng));

            Self {
                layers,
                w_final,
                b_final: 0.0,
                dropout_rate,
            }
        }

        pub fn forward(&self, sequence: &[f32], training: bool) -> f32 {
            let hidden_size = self.layers[0].hidden_size;
            let num_layers = self.layers.len();
            
            let mut h_states = vec![Array1::zeros(hidden_size); num_layers];
            let mut c_states = vec![Array1::zeros(hidden_size); num_layers];

            for i in (0..sequence.len()).step_by(4) {
                let mut layer_input = Array1::from_vec(vec![
                    sequence[i],     // Normalized closing price
                    sequence[i + 1], // Normalized opening price
                    sequence[i + 2], // Normalized moving average
                    sequence[i + 3], // Normalized momentum
                ]);
                
                for (j, layer) in self.layers.iter().enumerate() {
                    let (h_new, c_new) = layer.forward(&layer_input, &h_states[j], &c_states[j]);
                    h_states[j] = h_new.clone();
                    c_states[j] = c_new;
                    
                    layer_input = if training && self.dropout_rate > 0.0 {
                        &h_new * (1.0 - self.dropout_rate)
                    } else {
                        h_new
                    };
                }
            }

            self.w_final.dot(&h_states[num_layers - 1]) + self.b_final
        }

        pub fn train_step(&mut self, sequences: &[Vec<f32>], targets: &[f32], learning_rate: f32) -> f32 {
            let mut total_loss = 0.0;
            let batch_size = sequences.len();

            for (seq, &target) in sequences.iter().zip(targets.iter()) {
                let prediction = self.forward(seq, true);
                let loss = (prediction - target).powi(2);
                total_loss += loss;

                let error = prediction - target;
                let lr_scaled = learning_rate * 2.0 * error / batch_size as f32;

                let hidden_size = self.layers[0].hidden_size;
                let mut h_states = vec![Array1::zeros(hidden_size); self.layers.len()];
                let mut c_states = vec![Array1::zeros(hidden_size); self.layers.len()];

                for i in (0..seq.len()).step_by(4) {
                    let mut layer_input = Array1::from_vec(vec![seq[i], seq[i + 1], seq[i + 2], seq[i + 3]]);
                    
                    for (j, layer) in self.layers.iter().enumerate() {
                        let (h_new, c_new) = layer.forward(&layer_input, &h_states[j], &c_states[j]);
                        h_states[j] = h_new.clone();
                        c_states[j] = c_new;
                        layer_input = h_new;
                    }
                }

                let final_hidden = &h_states[self.layers.len() - 1];
                self.w_final = &self.w_final - &(final_hidden * lr_scaled);
                self.b_final -= lr_scaled;

                let gradient_scale = (lr_scaled * 0.01).max(-0.1).min(0.1);
                for layer in &mut self.layers {
                    for i in 0..layer.hidden_size {
                        layer.b_input[i] -= gradient_scale * 0.1;
                        layer.b_output[i] -= gradient_scale * 0.1;
                        layer.b_cell[i] -= gradient_scale * 0.1;
                        layer.b_forget[i] = (layer.b_forget[i] - gradient_scale * 0.05).max(0.1);
                    }
                }
            }

            total_loss / batch_size as f32
        }

        pub fn to_weights(&self, cli: &Cli, closing_mean: f32, closing_std: f32, opening_mean: f32, opening_std: f32) -> ModelWeights {
            ModelWeights {
                asset: String::new(), // Will be set in train_model
                source: "investing".to_string(), // Hardcoded to match original command
                layers: self.layers.iter().map(|layer| layer.to_weights()).collect(),
                w_final: self.w_final.clone(),
                b_final: self.b_final,
                closing_mean,
                closing_std,
                opening_mean,
                opening_std,
                seq_length: cli.seq_length,
                hidden_size: cli.hidden_size,
                num_layers: cli.num_layers,
                timestamp: Utc::now().to_rfc3339(),
            }
        }
    }
}

mod data {
    use super::{Client, StockData, LSTMError};
    use rand::seq::SliceRandom;
    use rand::rngs::ThreadRng;

    pub fn load_all_assets(pg_client: &mut Client) -> Result<Vec<String>, LSTMError> {
        let rows = pg_client.query(
            "SELECT DISTINCT asset FROM stock_records",
            &[],
        )?;
        
        let assets: Vec<String> = rows.into_iter().map(|row| row.get(0)).collect();
        Ok(assets)
    }

    pub fn load_data_from_postgres(pg_client: &mut Client, asset: &str) -> Result<Vec<StockData>, LSTMError> {
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

    pub fn normalize_data(prices: &[f32]) -> (Vec<f32>, f32, f32) {
        let mean = prices.iter().sum::<f32>() / prices.len() as f32;
        let variance = prices.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (prices.len() - 1) as f32;
        let std = variance.sqrt().max(1e-8);
        let normalized = prices.iter().map(|x| (x - mean) / std).collect();
        (normalized, mean, std)
    }

    pub fn create_sequences(data: &[StockData], seq_length: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
        let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
        
        let (norm_closing, closing_mean, closing_std) = normalize_data(&closing_prices);
        let (_norm_opening, _opening_mean, _opening_std) = normalize_data(&opening_prices);
        
        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        
        for i in 0..(data.len() - seq_length) {
            let mut sequence = Vec::new();
            for j in i..(i + seq_length) {
                sequence.push(norm_closing[j]);
                sequence.push(_norm_opening[j]);
                if j >= 4 {
                    let ma5 = closing_prices[(j-4)..=j].iter().sum::<f32>() / 5.0;
                    sequence.push((ma5 - closing_mean) / closing_std);
                } else {
                    sequence.push(norm_closing[j]);
                }
                sequence.push(if j > 0 { (closing_prices[j] - closing_prices[j-1]) / closing_std } else { 0.0 });
            }
            sequences.push(sequence);
            targets.push(norm_closing[i + seq_length]);
        }
        (sequences, targets)
    }

    pub fn create_batches(sequences: Vec<Vec<f32>>, targets: Vec<f32>, batch_size: usize) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut rng = ThreadRng::default();
        let mut combined: Vec<_> = sequences.into_iter().zip(targets).collect();
        combined.shuffle(&mut rng);
        
        combined.chunks(batch_size)
            .map(|chunk| {
                let (seqs, tgts): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                (seqs, tgts)
            })
            .collect()
    }
}

mod metrics {
    pub fn calculate_directional_accuracy(predictions: &[f32], targets: &[f32], sequences: &[Vec<f32>]) -> f32 {
        let mut correct = 0;
        let total = predictions.len();
        for i in 0..total {
            if let Some(&last_closing) = sequences[i].first() {
                let pred_direction = predictions[i] > last_closing;
                let actual_direction = targets[i] > last_closing;
                if pred_direction == actual_direction {
                    correct += 1;
                }
            }
        }
        correct as f32 / total as f32
    }

    pub fn calculate_mape(predictions: &[f32], targets: &[f32]) -> f32 {
        let mut total_percentage_error = 0.0;
        let mut count = 0;
        
        for (&pred, &actual) in predictions.iter().zip(targets.iter()) {
            if actual.abs() > 1e-8 {
                total_percentage_error += ((pred - actual) / actual).abs();
                count += 1;
            }
        }
        
        if count > 0 {
            (total_percentage_error / count as f32) * 100.0
        } else {
            0.0
        }
    }

    pub fn calculate_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
        let mse: f32 = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        mse.sqrt()
    }
}

mod storage {
    use super::{Cli, TrainingMetrics, LSTMError, Client, Json};
    use log::info;

    pub fn save_model_to_postgres(
        pg_client: &mut Client,
        model: &super::model::MultiLayerLSTM,
        cli: &Cli,
        asset: &str,
        closing_mean: f32,
        closing_std: f32,
        opening_mean: f32,
        opening_std: f32,
        metrics: &TrainingMetrics
    ) -> Result<(), LSTMError> {
        pg_client.execute(
            "CREATE TABLE IF NOT EXISTS lstm_weights_v3 (
                asset TEXT NOT NULL,
                source TEXT NOT NULL,
                weights_json JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (asset, source)
            )", &[])?;

        pg_client.execute(
            "CREATE TABLE IF NOT EXISTS training_metrics_v3 (
                asset TEXT NOT NULL,
                source TEXT NOT NULL,
                final_loss REAL,
                final_val_loss REAL,
                directional_accuracy REAL,
                mape REAL,
                rmse REAL,
                epochs_trained INTEGER,
                training_time DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (asset, source)
            )", &[])?;

        let mut weights = model.to_weights(cli, closing_mean, closing_std, opening_mean, opening_std);
        weights.asset = asset.to_string();

        pg_client.execute(
            "INSERT INTO lstm_weights_v3 (asset, source, weights_json) 
             VALUES ($1, $2, $3)
             ON CONFLICT ON CONSTRAINT lstm_weights_v3_pkey 
             DO UPDATE SET weights_json = $3, created_at = CURRENT_TIMESTAMP",
            &[&asset, &weights.source, &Json(&weights)]
        )?;

        pg_client.execute(
            "INSERT INTO training_metrics_v3 (asset, source, final_loss, final_val_loss, directional_accuracy, mape, rmse, epochs_trained, training_time) 
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             ON CONFLICT ON CONSTRAINT training_metrics_v3_pkey 
             DO UPDATE SET 
                 final_loss = $3, 
                 final_val_loss = $4, 
                 directional_accuracy = $5, 
                 mape = $6, 
                 rmse = $7, 
                 epochs_trained = $8, 
                 training_time = $9, 
                 created_at = CURRENT_TIMESTAMP",
            &[&metrics.asset, &metrics.source, &metrics.final_loss, &metrics.final_val_loss,
              &metrics.directional_accuracy, &metrics.mape, &metrics.rmse, &(metrics.epochs_trained as i32), &metrics.training_time]
        )?;

        info!("Enhanced model and metrics saved to PostgreSQL for asset {}", asset);
        Ok(())
    }
}

fn train_model(cli: Cli) -> Result<(), LSTMError> {
    let total_start_time = Instant::now();
    println!("🚀 Starting training for all assets (Enhanced Version)");

    let mut pg_client = Client::connect(&cli.pg_conn, NoTls)?;
    let assets = data::load_all_assets(&mut pg_client)?;
    println!("📊 Found {} unique assets to process", assets.len());

    let mut overall_metrics = Vec::new();

    for asset in assets {
        println!("\n🔍 Processing asset: {}", asset);
        let start_time = Instant::now();

        let data = data::load_data_from_postgres(&mut pg_client, &asset)?;
        if data.len() < cli.seq_length + 100 {
            println!("⚠️ Skipping asset {}: insufficient data ({} records, need {})", 
                    asset, data.len(), cli.seq_length + 100);
            continue;
        }
        println!("📊 Loaded {} records for asset {}", data.len(), asset);

        let (sequences, targets) = data::create_sequences(&data, cli.seq_length);
        let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
        let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
        let (_norm_closing, closing_mean, closing_std) = data::normalize_data(&closing_prices);
        let (_norm_opening, opening_mean, opening_std) = data::normalize_data(&opening_prices);
        
        info!("Data normalized for {} - Closing Mean: {:.4}, Closing Std: {:.4}, Opening Mean: {:.4}, Opening Std: {:.4}", 
              asset, closing_mean, closing_std, opening_mean, opening_std);
        info!("Created {} sequences of length {} (4 features per timestep) for {}", 
              sequences.len(), cli.seq_length, asset);

        let split_idx = (sequences.len() as f32 * cli.train_split) as usize;
        let train_sequences = sequences[..split_idx].to_vec();
        let train_targets = targets[..split_idx].to_vec();
        let val_sequences = sequences[split_idx..].to_vec();
        let val_targets = targets[split_idx..].to_vec();
        
        println!("🧠 Training: {} samples, Validation: {} samples for {}", 
                 train_sequences.len(), val_sequences.len(), asset);
        println!("🏗️ Model: {} layers, {} hidden units, {:.1}% dropout", 
                 cli.num_layers, cli.hidden_size, cli.dropout_rate * 100.0);

        let mut rng = ThreadRng::default();
        let input_size = 4; // Closing, opening, moving average, momentum
        let mut model = model::MultiLayerLSTM::new(input_size, cli.hidden_size, cli.num_layers, cli.dropout_rate, &mut rng);
        
        let mut best_val_loss = f32::INFINITY;
        let mut patience = 0;
        const MAX_PATIENCE: usize = 20;
        let mut lr = cli.learning_rate;

        println!("🎯 Starting enhanced training for {}...", asset);
        
        for epoch in 0..cli.epochs {
            let train_batches = data::create_batches(train_sequences.clone(), train_targets.clone(), cli.batch_size);
            let mut epoch_loss = 0.0;
            
            for (batch_seqs, batch_targets) in train_batches {
                let batch_loss = model.train_step(&batch_seqs, &batch_targets, lr);
                epoch_loss += batch_loss;
            }
            
            epoch_loss /= (train_sequences.len() / cli.batch_size) as f32;

            let mut val_loss = 0.0;
            let mut val_predictions = Vec::new();

            for (seq, &target) in val_sequences.iter().zip(val_targets.iter()) {
                let pred = model.forward(seq, false);
                val_predictions.push(pred);
                val_loss += (pred - target).powi(2);
            }
            val_loss /= val_sequences.len() as f32;

            let directional_acc = metrics::calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);
            
            if epoch > 0 && epoch % 30 == 0 {
                lr *= 0.9;
            }

            if epoch % 5 == 0 || epoch < 10 {
                println!("Epoch {:3}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.1}%, LR: {:.6}",
                        epoch + 1, epoch_loss, val_loss, directional_acc * 100.0, lr);
            }

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience = 0;
            } else {
                patience += 1;
                if patience >= MAX_PATIENCE {
                    println!("⏹️ Early stopping at epoch {} (best val loss: {:.6})", epoch + 1, best_val_loss);
                    break;
                }
            }

            if epoch_loss > 100.0 || epoch_loss.is_nan() {
                return Err(LSTMError::TrainingError(format!("Training diverged for asset {}", asset)));
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        println!("\n🔍 Final evaluation for {}...", asset);
        
        let mut train_predictions = Vec::new();
        let mut val_predictions = Vec::new();

        for seq in &train_sequences {
            train_predictions.push(model.forward(seq, false));
        }
        
        for seq in &val_sequences {
            val_predictions.push(model.forward(seq, false));
        }

        let final_train_loss = metrics::calculate_rmse(&train_predictions, &train_targets).powi(2);
        let final_val_loss = metrics::calculate_rmse(&val_predictions, &val_targets).powi(2);
        let train_directional_acc = metrics::calculate_directional_accuracy(&train_predictions, &train_targets, &train_sequences);
        let val_directional_acc = metrics::calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);
        
        let train_pred_denorm: Vec<f32> = train_predictions.iter().map(|x| x * closing_std + closing_mean).collect();
        let train_targets_denorm: Vec<f32> = train_targets.iter().map(|x| x * closing_std + closing_mean).collect();
        let val_pred_denorm: Vec<f32> = val_predictions.iter().map(|x| x * closing_std + closing_mean).collect();
        let val_targets_denorm: Vec<f32> = val_targets.iter().map(|x| x * closing_std + closing_mean).collect();
        
        let train_mape = metrics::calculate_mape(&train_pred_denorm, &train_targets_denorm);
        let val_mape = metrics::calculate_mape(&val_pred_denorm, &val_targets_denorm);
        let train_rmse = metrics::calculate_rmse(&train_pred_denorm, &train_targets_denorm);
        let val_rmse = metrics::calculate_rmse(&val_pred_denorm, &val_targets_denorm);

        let training_metrics = TrainingMetrics {
            asset: asset.clone(),
            source: "investing".to_string(), // Hardcoded to match original command
            final_loss: final_train_loss,
            final_val_loss,
            directional_accuracy: val_directional_acc,
            mape: val_mape,
            rmse: val_rmse,
            epochs_trained: cli.epochs - patience,
            training_time,
            timestamp: Utc::now().to_rfc3339(),
        };

        println!("💾 Saving enhanced model for {} to PostgreSQL...", asset);
        storage::save_model_to_postgres(&mut pg_client, &model, &cli, &asset, closing_mean, closing_std, opening_mean, opening_std, &training_metrics)?;

        println!("\n✅ Training completed for {}!", asset);
        println!("   📊 Final Results:");
        println!("      🎯 Train Loss: {:.6}", final_train_loss);
        println!("      🎯 Val Loss: {:.6}", final_val_loss);
        println!("      📈 Train Dir Accuracy: {:.1}%", train_directional_acc * 100.0);
        println!("      📈 Val Dir Accuracy: {:.1}%", val_directional_acc * 100.0);
        println!("      📊 Train MAPE: {:.2}%", train_mape);
        println!("      📊 Val MAPE: {:.2}%", val_mape);
        println!("      📏 Train RMSE: {:.4}", train_rmse);
        println!("      📏 Val RMSE: {:.4}", val_rmse);
        println!("      ⏱️ Training Time: {:.1}s", training_time);
        println!("      🔢 Data Stats: Closing Mean={:.2}, Closing Std={:.2}, Opening Mean={:.2}, Opening Std={:.2}", 
                 closing_mean, closing_std, opening_mean, opening_std);

        println!("\n🔮 Sample predictions for {} (last 5 validation samples):", asset);
        let n_val = val_sequences.len();
        for i in (n_val.saturating_sub(5))..n_val {
            let pred = model.forward(&val_sequences[i], false);
            let actual = val_targets[i];
            let last_closing = val_sequences[i].first().unwrap_or(&0.0);
            let pred_denorm = pred * closing_std + closing_mean;
            let actual_denorm = actual * closing_std + closing_mean;
            let last_denorm = last_closing * closing_std + closing_mean;
            
            let direction_correct = (pred > *last_closing) == (actual > *last_closing);
            let direction_indicator = if direction_correct { "✅" } else { "❌" };
            
            println!("      Sample {}: Last Closing={:.2} → Pred={:.2}, Actual={:.2} (Error: {:.2}) {}",
                    i + 1, last_denorm, pred_denorm, actual_denorm, 
                    (pred_denorm - actual_denorm).abs(), direction_indicator);
        }

        overall_metrics.push((asset, val_directional_acc, val_mape, val_rmse));
    }

    let total_training_time = total_start_time.elapsed().as_secs_f64();
    println!("\n📊 Summary for all assets:");
    println!("   ⏱️ Total Training Time: {:.1}s", total_training_time);
    println!("   🔢 Assets Processed: {}", overall_metrics.len());
    
    let avg_directional_acc = overall_metrics.iter().map(|(_, acc, _, _)| acc).sum::<f32>() / overall_metrics.len() as f32;
    let avg_mape = overall_metrics.iter().map(|(_, _, mape, _)| mape).sum::<f32>() / overall_metrics.len() as f32;
    let avg_rmse = overall_metrics.iter().map(|(_, _, _, rmse)| rmse).sum::<f32>() / overall_metrics.len() as f32;

    println!("   📈 Average Directional Accuracy: {:.1}%", avg_directional_acc * 100.0);
    println!("   📊 Average MAPE: {:.2}%", avg_mape);
    println!("   📏 Average RMSE: {:.4}", avg_rmse);

    println!("\n📈 Per-Asset Performance:");
    for (asset, acc, mape, rmse) in overall_metrics {
        println!("      {}: Dir Acc={:.1}%, MAPE={:.2}%, RMSE={:.4}", asset, acc * 100.0, mape, rmse);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();
    info!("Starting Enhanced LSTM training for all assets with parameters:");
    info!("  Sequence Length: {}", cli.seq_length);
    info!("  Hidden Size: {}", cli.hidden_size);
    info!("  Number of Layers: {}", cli.num_layers);
    info!("  Learning Rate: {}", cli.learning_rate);
    info!("  Epochs: {}", cli.epochs);
    info!("  Batch Size: {}", cli.batch_size);
    info!("  Dropout Rate: {}", cli.dropout_rate);

    train_model(cli).map_err(|e| {
        error!("Enhanced training failed: {}", e);
        Box::new(e) as Box<dyn std::error::Error>
    })
}



// cargo run -- --source investing --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 150 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005

// cargo run -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 5 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005



// # Conservative configuration (stable)
// cargo run -- --asset WEGE3 --source investing --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 150 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005

// # Aggressive configuration (higher capacity)
// cargo run -- --asset WEGE3 --source investing --seq-length 50 --hidden-size 256 --num-layers 3 --epochs 200 --batch-size 32 --dropout-rate 0.2 --learning-rate 0.001
 

// # Configuração conservadora (mais estável)
// cargo run -- --asset WEGE3 --source investing  --seq-length 40   --hidden-size 64   --num-layers 2   --epochs 150   --batch-size 16   --dropout-rate 0.3   --learning-rate 0.0005

// # Configuração agressiva (mais capacidade)
// cargo run -- --asset WEGE3 --source investing --seq-length 50  --hidden-size 256  --num-layers 3   --epochs 2    --batch-size 32   --dropout-rate 0.2  --learning-rate 0.001

// Exemplo de uso:
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 64 --epochs 2 --learning-rate 0.001
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 64 --epochs 2 --learning-rate 0.001

// cargo run -- --asset WEGE3 --source investing --seq-length 50 --hidden-size 256 --num-layers 3 --epochs 2 --batch-size 32 --dropout-rate 0.2 --learning-rate 0.001
// cargo run -- --asset WEGE3 --source investing --seq-length 50 --hidden-size 256 --num-layers 3 --epochs 2 --batch-size 32 --dropout-rate 0.2 --learning-rate 0.001


// Uso:
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50 --epochs 100
 
 
  

// Example usage:
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50 --learning-rate 0.001 --epochs 100 --batch-size 32
// cargo run -- --asset WEGE3 --source investing --seq-length 30 --hidden-size 64 --learning-rate 0.0005 --epochs 200 --batch-size 16 --skip-training-data

 
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50 --learning-rate 0.001 --epochs 100



// cd lstmfiletrain
// cargo run -- --asset WEGE3 --source investing

// # 2. Train the LSTM model
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
 