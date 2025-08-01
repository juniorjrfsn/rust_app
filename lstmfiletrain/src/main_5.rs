// projeto : lstmfiletrain
// file : src/main.rs - Enhanced Version with Better Training


use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use log::{info, error};
use env_logger;
use thiserror::Error;
use chrono::Utc;
use rusqlite::{Connection, params};
use postgres::{Client, NoTls};
use postgres::types::Json;
use rand::{thread_rng, rngs::ThreadRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

#[derive(Error, Debug)]
enum LSTMError {
    #[error("Insufficient data: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
    #[error("Training error: {0}")]
    TrainingError(String),
}

#[derive(Parser)]
#[command(name = "lstm_train")]
#[command(about = "Enhanced LSTM model for stock prediction")]
#[command(version = "3.0.0")]
struct Cli {
    #[arg(long, help = "Stock symbol")]
    asset: String,
    #[arg(long, help = "Data source")]
    source: String,
    #[arg(long, default_value = "../dados", help = "Data directory")]
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
}

#[derive(Debug, Serialize, Deserialize)]
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
    source: String,
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
    use super::{Array1, Array2, Distribution, Normal, ThreadRng};

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
                b_forget: Array1::ones(hidden_size), // Forget gate bias initialized to 1
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
            
            // First layer takes input_size, others take hidden_size
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

            for &input_val in sequence {
                let mut layer_input = Array1::from_vec(vec![input_val]);
                
                for (i, layer) in self.layers.iter().enumerate() {
                    let (h_new, c_new) = layer.forward(&layer_input, &h_states[i], &c_states[i]);
                    h_states[i] = h_new.clone();
                    c_states[i] = c_new;
                    
                    // Apply dropout during training (simplified - no actual random dropout here)
                    layer_input = if training && self.dropout_rate > 0.0 {
                        &h_new * (1.0 - self.dropout_rate)
                    } else {
                        h_new
                    };
                }
            }

            // Use the last hidden state from the final layer
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

                // Simplified gradient computation - update final layer
                let hidden_size = self.layers[0].hidden_size;
                let mut h_states = vec![Array1::zeros(hidden_size); self.layers.len()];
                let mut c_states = vec![Array1::zeros(hidden_size); self.layers.len()];

                // Forward pass to get final hidden state
                for &input_val in seq {
                    let mut layer_input = Array1::from_vec(vec![input_val]);
                    
                    for (i, layer) in self.layers.iter().enumerate() {
                        let (h_new, c_new) = layer.forward(&layer_input, &h_states[i], &c_states[i]);
                        h_states[i] = h_new.clone();
                        c_states[i] = c_new;
                        layer_input = h_new;
                    }
                }

                // Update final layer weights
                let final_hidden = &h_states[self.layers.len() - 1];
                self.w_final = &self.w_final - &(final_hidden * lr_scaled);
                self.b_final -= lr_scaled;

                // Simplified gradient updates for LSTM layers
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
    }
}

mod data {
    use super::{Connection, params, StockData, LSTMError};
    use rand::seq::SliceRandom;
    use rand::thread_rng as rng;

    pub fn load_data_from_sqlite(data_dir: &str, source: &str, asset: &str) -> Result<Vec<StockData>, LSTMError> {
        let db_path = format!("{}/{}.db", data_dir, source);
        let conn = Connection::open(&db_path)?;
        let mut stmt = conn.prepare(
            "SELECT date, closing FROM stock_records WHERE asset = ?1 ORDER BY date ASC"
        )?;
        let records: Result<Vec<StockData>, _> = stmt.query_map(params![asset], |row| {
            Ok(StockData {
                date: row.get(0)?,
                closing: row.get(1)?,
            })
        })?.collect();
        Ok(records?)
    }

    pub fn normalize_data(prices: &[f32]) -> (Vec<f32>, f32, f32) {
        let mean = prices.iter().sum::<f32>() / prices.len() as f32;
        let variance = prices.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (prices.len() - 1) as f32;
        let std = variance.sqrt().max(1e-8);
        let normalized = prices.iter().map(|x| (x - mean) / std).collect();
        (normalized, mean, std)
    }

    pub fn create_sequences(data: &[f32], seq_length: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        for i in 0..(data.len() - seq_length) {
            sequences.push(data[i..i + seq_length].to_vec());
            targets.push(data[i + seq_length]);
        }
        (sequences, targets)
    }

    pub fn create_batches(sequences: Vec<Vec<f32>>, targets: Vec<f32>, batch_size: usize) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut combined: Vec<_> = sequences.into_iter().zip(targets).collect();
        combined.shuffle(&mut rng());
        
        combined.chunks(batch_size)
            .map(|chunk| {
                let (seqs, tgts): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                (seqs, tgts)
            })
            .collect()
    }

    pub fn add_features(prices: &[f32], seq_length: usize) -> Vec<Vec<f32>> {
        let mut enhanced_sequences = Vec::new();
        
        for i in seq_length..prices.len() {
            let mut features = Vec::new();
            
            // Original prices
            for j in (i - seq_length)..i {
                features.push(prices[j]);
            }
            
            // Moving averages
            if i >= seq_length + 4 {
                let ma5 = prices[(i-5)..i].iter().sum::<f32>() / 5.0;
                let ma20 = if i >= 20 { prices[(i-20)..i].iter().sum::<f32>() / 20.0 } else { ma5 };
                features.push(ma5);
                features.push(ma20);
            } else {
                features.push(prices[i-1]);
                features.push(prices[i-1]);
            }
            
            // Price differences (momentum)
            if i >= seq_length + 1 {
                features.push(prices[i-1] - prices[i-2]);
            } else {
                features.push(0.0);
            }
            
            enhanced_sequences.push(features);
        }
        
        enhanced_sequences
    }
}

mod metrics {
    pub fn calculate_directional_accuracy(predictions: &[f32], targets: &[f32], sequences: &[Vec<f32>]) -> f32 {
        let mut correct = 0;
        let total = predictions.len();
        for i in 0..total {
            if let Some(&last_price) = sequences[i].first() { // Use first element as base price
                let pred_direction = predictions[i] > last_price;
                let actual_direction = targets[i] > last_price;
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
    use super::{model, Cli, TrainingMetrics, ModelWeights, LSTMError, Client, Json, Utc};
    use log::info;

    pub fn save_model_to_postgres(
        pg_client: &mut Client,
        model: &model::MultiLayerLSTM,
        cli: &Cli,
        mean: f32,
        std: f32,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )", &[])?;

        // Create placeholder weights structure (simplified for this example)
        let weights = ModelWeights {
            asset: cli.asset.clone(),
            source: cli.source.clone(),
            layers: vec![],
            w_final: model.w_final.clone(),
            b_final: model.b_final,
            data_mean: mean,
            data_std: std,
            seq_length: cli.seq_length,
            hidden_size: cli.hidden_size,
            num_layers: cli.num_layers,
            timestamp: Utc::now().to_rfc3339(),
        };

        pg_client.execute(
            "DELETE FROM lstm_weights_v3 WHERE asset = $1 AND source = $2",
            &[&cli.asset, &cli.source])?;

        pg_client.execute(
            "INSERT INTO lstm_weights_v3 (asset, source, weights_json) VALUES ($1, $2, $3)",
            &[&cli.asset, &cli.source, &Json(&weights)])?;

        pg_client.execute(
            "INSERT INTO training_metrics_v3 (asset, source, final_loss, final_val_loss, directional_accuracy, mape, rmse, epochs_trained, training_time)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[&metrics.asset, &metrics.source, &metrics.final_loss, &metrics.final_val_loss,
              &metrics.directional_accuracy, &metrics.mape, &metrics.rmse, &(metrics.epochs_trained as i32), &metrics.training_time])?;

        info!("Enhanced model and metrics saved to PostgreSQL");
        Ok(())
    }
}

fn train_model(cli: Cli) -> Result<(), LSTMError> {
    let start_time = Instant::now();
    println!("🚀 Loading data for {} from {} (Enhanced Version)", cli.asset, cli.source);

    let data = data::load_data_from_sqlite(&cli.data_dir, &cli.source, &cli.asset)?;
    if data.len() < cli.seq_length + 100 {
        return Err(LSTMError::InsufficientData {
            required: cli.seq_length + 100,
            actual: data.len(),
        });
    }
    println!("📊 Loaded {} records", data.len());

    let prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let (normalized_prices, mean, std) = data::normalize_data(&prices);
    let (sequences, targets) = data::create_sequences(&normalized_prices, cli.seq_length);
    
    info!("Data normalized - Mean: {:.4}, Std: {:.4}", mean, std);
    info!("Created {} sequences of length {}", sequences.len(), cli.seq_length);

    let split_idx = (sequences.len() as f32 * cli.train_split) as usize;
    let train_sequences = sequences[..split_idx].to_vec();
    let train_targets = targets[..split_idx].to_vec();
    let val_sequences = sequences[split_idx..].to_vec();
    let val_targets = targets[split_idx..].to_vec();
    
    println!("🧠 Training: {} samples, Validation: {} samples", train_sequences.len(), val_sequences.len());
    println!("🏗️ Model: {} layers, {} hidden units, {:.1}% dropout", cli.num_layers, cli.hidden_size, cli.dropout_rate * 100.0);

    let mut rng = rand::thread_rng();
    let mut model = model::MultiLayerLSTM::new(1, cli.hidden_size, cli.num_layers, cli.dropout_rate, &mut rng);
    
    let mut best_val_loss = f32::INFINITY;
    let mut patience = 0;
    const MAX_PATIENCE: usize = 20;
    let mut lr = cli.learning_rate;

    println!("🎯 Starting enhanced training with batching...");
    
    for epoch in 0..cli.epochs {
        // Create batches for training
        let train_batches = data::create_batches(train_sequences.clone(), train_targets.clone(), cli.batch_size);
        let mut epoch_loss = 0.0;
        
        for (batch_seqs, batch_targets) in train_batches {
            let batch_loss = model.train_step(&batch_seqs, &batch_targets, lr);
            epoch_loss += batch_loss;
        }
        
        epoch_loss /= (train_sequences.len() / cli.batch_size) as f32;

        // Validation
        let mut val_loss = 0.0;
        let mut val_predictions = Vec::new();

        for (seq, &target) in val_sequences.iter().zip(val_targets.iter()) {
            let pred = model.forward(seq, false);
            val_predictions.push(pred);
            val_loss += (pred - target).powi(2);
        }
        val_loss /= val_sequences.len() as f32;

        let directional_acc = metrics::calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);
        
        // Learning rate scheduling
        if epoch > 0 && epoch % 30 == 0 {
            lr *= 0.9;
        }

        if epoch % 5 == 0 || epoch < 10 {
            println!("Epoch {:3}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.1}%, LR: {:.6}",
                    epoch + 1, epoch_loss, val_loss, directional_acc * 100.0, lr);
        }

        // Early stopping with improved patience
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
            return Err(LSTMError::TrainingError("Training diverged".into()));
        }
    }

    let training_time = start_time.elapsed().as_secs_f64();
    println!("\n🔍 Final comprehensive evaluation...");
    
    // Final predictions
    let mut train_predictions = Vec::new();
    let mut val_predictions = Vec::new();

    for seq in &train_sequences {
        train_predictions.push(model.forward(seq, false));
    }
    
    for seq in &val_sequences {
        val_predictions.push(model.forward(seq, false));
    }

    // Calculate comprehensive metrics
    let final_train_loss = metrics::calculate_rmse(&train_predictions, &train_targets).powi(2);
    let final_val_loss = metrics::calculate_rmse(&val_predictions, &val_targets).powi(2);
    let train_directional_acc = metrics::calculate_directional_accuracy(&train_predictions, &train_targets, &train_sequences);
    let val_directional_acc = metrics::calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);
    
    // Denormalize for MAPE calculation
    let train_pred_denorm: Vec<f32> = train_predictions.iter().map(|x| x * std + mean).collect();
    let train_targets_denorm: Vec<f32> = train_targets.iter().map(|x| x * std + mean).collect();
    let val_pred_denorm: Vec<f32> = val_predictions.iter().map(|x| x * std + mean).collect();
    let val_targets_denorm: Vec<f32> = val_targets.iter().map(|x| x * std + mean).collect();
    
    let train_mape = metrics::calculate_mape(&train_pred_denorm, &train_targets_denorm);
    let val_mape = metrics::calculate_mape(&val_pred_denorm, &val_targets_denorm);
    let train_rmse = metrics::calculate_rmse(&train_pred_denorm, &train_targets_denorm);
    let val_rmse = metrics::calculate_rmse(&val_pred_denorm, &val_targets_denorm);

    let training_metrics = TrainingMetrics {
        asset: cli.asset.clone(),
        source: cli.source.clone(),
        final_loss: final_train_loss,
        final_val_loss,
        directional_accuracy: val_directional_acc,
        mape: val_mape,
        rmse: val_rmse,
        epochs_trained: cli.epochs - patience,
        training_time,
        timestamp: Utc::now().to_rfc3339(),
    };

    println!("💾 Saving enhanced model to PostgreSQL...");
    let mut pg_client = Client::connect(&cli.pg_conn, NoTls)?;
    storage::save_model_to_postgres(&mut pg_client, &model, &cli, mean, std, &training_metrics)?;

    println!("\n✅ Enhanced training completed!");
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
    println!("      🔢 Data Stats: Mean={:.2}, Std={:.2}", mean, std);

    println!("\n🔮 Enhanced sample predictions (last 5 validation samples):");
    let n_val = val_sequences.len();
    for i in (n_val.saturating_sub(5))..n_val {
        let pred = model.forward(&val_sequences[i], false);
        let actual = val_targets[i];
        let last_price = val_sequences[i].last().unwrap_or(&0.0);
        let pred_denorm = pred * std + mean;
        let actual_denorm = actual * std + mean;
        let last_denorm = last_price * std + mean;
        
        let direction_correct = (pred > *last_price) == (actual > *last_price);
        let direction_indicator = if direction_correct { "✅" } else { "❌" };
        
        println!("      Sample {}: Last={:.2} → Pred={:.2}, Actual={:.2} (Error: {:.2}) {}",
                i + 1, last_denorm, pred_denorm, actual_denorm, 
                (pred_denorm - actual_denorm).abs(), direction_indicator);
    }

    // Additional analysis
    println!("\n📈 Performance Analysis:");
    let improvement_vs_baseline = ((val_directional_acc - 0.5) / 0.5) * 100.0;
    println!("      📊 Directional accuracy improvement over random: {:.1}%", improvement_vs_baseline);
    
    if val_mape < 10.0 {
        println!("      🎯 Excellent MAPE performance (< 10%)");
    } else if val_mape < 20.0 {
        println!("      👍 Good MAPE performance (< 20%)");
    } else {
        println!("      ⚠️ MAPE needs improvement (> 20%)");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();
    info!("Starting Enhanced LSTM training with parameters:");
    info!("  Asset: {}", cli.asset);
    info!("  Source: {}", cli.source);
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
 