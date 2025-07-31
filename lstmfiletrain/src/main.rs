// projeto : lstmfiletrain
// file : src/main.rs


use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use log::error;
use env_logger;
use thiserror::Error;
use chrono::Utc;
use rusqlite::{Connection, params};
use postgres::{Client, NoTls};
use postgres::types::Json;
use rand::thread_rng;
use rand_distr::{Distribution, Uniform};

#[allow(dead_code)]
#[derive(Error, Debug)]
enum LSTMError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Insufficient data: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[derive(Parser)]
#[command(name = "lstm_train")]
#[command(about = "Train LSTM model on stock data")]
#[command(version = "2.0.0")]
struct Cli {
    #[arg(long, help = "Stock symbol")]
    asset: String,
    #[arg(long, help = "Data source")]
    source: String,
    #[arg(long, default_value = "../dados", help = "Data directory")]
    data_dir: String,
    #[arg(long, default_value_t = 20, help = "Sequence length")]
    seq_length: usize,
    #[arg(long, default_value_t = 0.001, help = "Learning rate")]
    learning_rate: f32,
    #[arg(long, default_value_t = 100, help = "Training epochs")]
    epochs: usize,
    #[arg(long, default_value_t = 50, help = "Hidden layer size")]
    hidden_size: usize,
    #[arg(long, default_value_t = 0.8, help = "Train/validation split")]
    train_split: f32,
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
    // Input gate weights
    w_input: Array2<f32>,
    b_input: Array1<f32>,
    // Forget gate weights  
    w_forget: Array2<f32>,
    b_forget: Array1<f32>,
    // Output gate weights
    w_output: Array2<f32>,
    b_output: Array1<f32>,
    // Cell state weights
    w_cell: Array2<f32>,
    b_cell: Array1<f32>,
    // Final output layer
    w_final: Array1<f32>,
    b_final: f32,
    // Normalization parameters
    mean: f32,
    std: f32,
    seq_length: usize,
    hidden_size: usize,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingMetrics {
    asset: String,
    source: String,
    final_loss: f32,
    accuracy: f32,
    epochs_trained: usize,
    training_time: f64,
    timestamp: String,
}

struct SimpleLSTM {
    hidden_size: usize,
    // Gate weights (input_size + hidden_size, hidden_size)
    w_input: Array2<f32>,
    w_forget: Array2<f32>, 
    w_output: Array2<f32>,
    w_cell: Array2<f32>,
    // Gate biases
    b_input: Array1<f32>,
    b_forget: Array1<f32>,
    b_output: Array1<f32>,
    b_cell: Array1<f32>,
    // Output layer
    w_final: Array1<f32>,
    b_final: f32,
}

impl SimpleLSTM {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let uniform = Uniform::new(-scale, scale).expect("Failed to create Uniform distribution");
        let uniform_small = Uniform::new(-0.1f32, 0.1f32).expect("Failed to create Uniform distribution");
        
        // Initialize weights directly
        let w_input = Array2::from_shape_fn((hidden_size, input_size + hidden_size), 
            |_| uniform.sample(&mut rng));
        let w_forget = Array2::from_shape_fn((hidden_size, input_size + hidden_size), 
            |_| uniform.sample(&mut rng));
        let w_output = Array2::from_shape_fn((hidden_size, input_size + hidden_size), 
            |_| uniform.sample(&mut rng));
        let w_cell = Array2::from_shape_fn((hidden_size, input_size + hidden_size), 
            |_| uniform.sample(&mut rng));
        
        let b_input = Array1::from_shape_fn(hidden_size, |_| uniform_small.sample(&mut rng));
        let b_output = Array1::from_shape_fn(hidden_size, |_| uniform_small.sample(&mut rng));
        let b_cell = Array1::from_shape_fn(hidden_size, |_| uniform_small.sample(&mut rng));
        let w_final = Array1::from_shape_fn(hidden_size, |_| uniform_small.sample(&mut rng));
        
        Self {
            hidden_size,
            w_input,
            w_forget,
            w_output,
            w_cell,
            b_input,
            b_forget: Array1::ones(hidden_size), // Forget gate bias = 1
            b_output,
            b_cell,
            w_final,
            b_final: 0.0,
        }
    }
    
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-10.0, 10.0)).exp())
    }
    
    fn tanh(x: f32) -> f32 {
        x.clamp(-10.0, 10.0).tanh()
    }
    
    fn forward(&self, sequence: &[f32]) -> f32 {
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        
        for &input in sequence {
            // Concatenate input and hidden state
            let mut x_h = Array1::zeros(1 + self.hidden_size);
            x_h[0] = input;
            for i in 0..self.hidden_size {
                x_h[1 + i] = h[i];
            }
            
            // Compute gates
            let i_gate = (&self.w_input.dot(&x_h) + &self.b_input).mapv(Self::sigmoid);
            let f_gate = (&self.w_forget.dot(&x_h) + &self.b_forget).mapv(Self::sigmoid);
            let o_gate = (&self.w_output.dot(&x_h) + &self.b_output).mapv(Self::sigmoid);
            let g_gate = (&self.w_cell.dot(&x_h) + &self.b_cell).mapv(Self::tanh);
            
            // Update cell state and hidden state
            c = &f_gate * &c + &i_gate * &g_gate;
            h = &o_gate * &c.mapv(Self::tanh);
        }
        
        // Final output
        self.w_final.dot(&h) + self.b_final
    }
    
    fn backward(&mut self, sequences: &[Vec<f32>], targets: &[f32], lr: f32) -> f32 {
        let mut total_loss = 0.0;
        let batch_size = sequences.len() as f32;
        
        // Simple gradient descent (simplified backpropagation)
        for (seq, &target) in sequences.iter().zip(targets.iter()) {
            let prediction = self.forward(seq);
            let loss = (prediction - target).powi(2);
            total_loss += loss;
            
            // Simple gradient update (approximation)
            let error = prediction - target;
            let grad_scale = 2.0 * error * lr / batch_size;
            
            // Update output weights
            let mut h = Array1::zeros(self.hidden_size);
            let mut c = Array1::zeros(self.hidden_size);
            
            // Forward pass to get final hidden state
            for &input in seq {
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
            
            // Update final layer
            self.w_final -= &(&h * grad_scale);
            self.b_final -= grad_scale;
        }
        
        total_loss / batch_size
    }
}

fn load_data_from_sqlite(data_dir: &str, source: &str, asset: &str) -> Result<Vec<StockData>, LSTMError> {
    let db_path = format!("{}/{}.db", data_dir, source);
    let conn = Connection::open(&db_path)?;
    
    let mut stmt = conn.prepare(
        "SELECT date, closing FROM stock_records WHERE asset = ?1 ORDER BY date"
    )?;
    
    let records: Result<Vec<StockData>, _> = stmt.query_map(params![asset], |row| {
        Ok(StockData {
            date: row.get(0)?,
            closing: row.get(1)?,
        })
    })?.collect();
    
    Ok(records?)
}

fn normalize_data(prices: &[f32]) -> (Vec<f32>, f32, f32) {
    let mean = prices.iter().sum::<f32>() / prices.len() as f32;
    let variance = prices.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / prices.len() as f32;
    let std = variance.sqrt();
    
    let normalized = prices.iter().map(|x| (x - mean) / std).collect();
    (normalized, mean, std)
}

fn create_sequences(data: &[f32], seq_length: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..(data.len() - seq_length) {
        sequences.push(data[i..i + seq_length].to_vec());
        targets.push(data[i + seq_length]);
    }
    
    (sequences, targets)
}

fn save_model_to_postgres(
    pg_client: &mut Client, 
    model: &SimpleLSTM, 
    cli: &Cli,
    mean: f32,
    std: f32,
    metrics: &TrainingMetrics
) -> Result<(), LSTMError> {
    // Create tables
    pg_client.execute(
        "CREATE TABLE IF NOT EXISTS lstm_weights (
            asset TEXT NOT NULL,
            source TEXT NOT NULL,
            weights_json JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset, source)
        )", &[])?;
    
    pg_client.execute(
        "CREATE TABLE IF NOT EXISTS training_metrics (
            asset TEXT NOT NULL,
            source TEXT NOT NULL,
            final_loss REAL,
            accuracy REAL,
            epochs_trained INTEGER,
            training_time DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )", &[])?;
    
    // Save model weights
    let weights = ModelWeights {
        asset: cli.asset.clone(),
        source: cli.source.clone(),
        w_input: model.w_input.clone(),
        b_input: model.b_input.clone(),
        w_forget: model.w_forget.clone(),
        b_forget: model.b_forget.clone(),
        w_output: model.w_output.clone(),
        b_output: model.b_output.clone(),
        w_cell: model.w_cell.clone(),
        b_cell: model.b_cell.clone(),
        w_final: model.w_final.clone(),
        b_final: model.b_final,
        mean,
        std,
        seq_length: cli.seq_length,
        hidden_size: cli.hidden_size,
        timestamp: Utc::now().to_rfc3339(),
    };
    
    // Delete existing model
    pg_client.execute(
        "DELETE FROM lstm_weights WHERE asset = $1 AND source = $2",
        &[&cli.asset, &cli.source])?;
    
    // Insert new model using Json wrapper
    pg_client.execute(
        "INSERT INTO lstm_weights (asset, source, weights_json) VALUES ($1, $2, $3)",
        &[&cli.asset, &cli.source, &Json(&weights)])?;
    
    // Save metrics  
    pg_client.execute(
        "INSERT INTO training_metrics (asset, source, final_loss, accuracy, epochs_trained, training_time)
         VALUES ($1, $2, $3, $4, $5, $6)",
        &[&metrics.asset, &metrics.source, &metrics.final_loss, &metrics.accuracy, 
          &(metrics.epochs_trained as i32), &metrics.training_time])?;
    
    Ok(())
}

fn train_model(cli: Cli) -> Result<(), LSTMError> {
    let start_time = std::time::Instant::now();
    
    println!("🚀 Loading data for {} from {}", cli.asset, cli.source);
    
    // Load data from SQLite
    let data = load_data_from_sqlite(&cli.data_dir, &cli.source, &cli.asset)?;
    
    if data.len() < cli.seq_length + 10 {
        return Err(LSTMError::InsufficientData {
            required: cli.seq_length + 10,
            actual: data.len(),
        });
    }
    
    println!("📊 Loaded {} records", data.len());
    
    // Prepare data
    let prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let (normalized_prices, mean, std) = normalize_data(&prices);
    let (sequences, targets) = create_sequences(&normalized_prices, cli.seq_length);
    
    // Split data
    let split_idx = (sequences.len() as f32 * cli.train_split) as usize;
    let (train_seqs, val_seqs) = sequences.split_at(split_idx);
    let (train_targets, val_targets) = targets.split_at(split_idx);
    
    println!("🧠 Training: {} samples, Validation: {} samples", train_seqs.len(), val_seqs.len());
    
    // Initialize model
    let mut model = SimpleLSTM::new(1, cli.hidden_size);
    
    // Training loop
    let mut best_loss = f32::INFINITY;
    let mut patience = 0;
    const MAX_PATIENCE: usize = 10;
    
    for epoch in 0..cli.epochs {
        let train_loss = model.backward(train_seqs, train_targets, cli.learning_rate);
        
        // Validation
        let mut val_loss = 0.0;
        let mut correct = 0;
        
        for (seq, &target) in val_seqs.iter().zip(val_targets.iter()) {
            let pred = model.forward(seq);
            val_loss += (pred - target).powi(2);
            
            // Direction accuracy
            if seq.len() > 1 {
                let last = seq[seq.len() - 1];
                if (pred > last) == (target > last) {
                    correct += 1;
                }
            }
        }
        
        val_loss /= val_seqs.len() as f32;
        let accuracy = correct as f32 / val_seqs.len() as f32;
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Train Loss: {:.6}, Val Loss: {:.6}, Accuracy: {:.2}%", 
                    epoch + 1, train_loss, val_loss, accuracy * 100.0);
        }
        
        // Early stopping
        if val_loss < best_loss {
            best_loss = val_loss;
            patience = 0;
        } else {
            patience += 1;
            if patience >= MAX_PATIENCE {
                println!("⏹️  Early stopping at epoch {}", epoch + 1);
                break;
            }
        }
    }
    
    let training_time = start_time.elapsed().as_secs_f64();
    
    // Final validation
    let mut final_loss = 0.0;
    let mut correct = 0;
    
    for (seq, &target) in val_seqs.iter().zip(val_targets.iter()) {
        let pred = model.forward(seq);
        final_loss += (pred - target).powi(2);
        if seq.len() > 1 {
            let last = seq[seq.len() - 1];
            if (pred > last) == (target > last) {
                correct += 1;
            }
        }
    }
    
    final_loss /= val_seqs.len() as f32;
    let final_accuracy = correct as f32 / val_seqs.len() as f32;
    
    let metrics = TrainingMetrics {
        asset: cli.asset.clone(),
        source: cli.source.clone(),
        final_loss,
        accuracy: final_accuracy,
        epochs_trained: cli.epochs.min(patience + 1),
        training_time,
        timestamp: Utc::now().to_rfc3339(),
    };
    
    // Save to PostgreSQL
    println!("💾 Saving model to PostgreSQL...");
    let mut pg_client = Client::connect(&cli.pg_conn, NoTls)?;
    save_model_to_postgres(&mut pg_client, &model, &cli, mean, std, &metrics)?;
    
    println!("✅ Training complete!");
    println!("   🎯 Final Loss: {:.6}", final_loss);
    println!("   📈 Accuracy: {:.2}%", final_accuracy * 100.0);
    println!("   ⏱️  Time: {:.2}s", training_time);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    
    train_model(cli).map_err(|e| {
        error!("Training failed: {}", e);
        Box::new(e) as Box<dyn std::error::Error>
    })
}

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
 