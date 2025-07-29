

use clap::{Parser, Subcommand};
use std::fs::{File, OpenOptions};
use std::path::Path;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use toml;
use serde_json;
use ndarray::{Array1, Array2, s};
use log::{info, warn, error};
use env_logger;
use rand::Rng;
use thiserror::Error;
use chrono::{Duration, NaiveDate};

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
enum LSTMError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    #[error("Invalid CSV format: {msg}")]
    InvalidCsv { msg: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Insufficient data: need at least {required} records, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ============================================================================
// CLI STRUCTURE
// ============================================================================

#[derive(Parser)]
#[command(name = "lstm-stock-predictor")]
#[command(about = "Complete LSTM pipeline for stock price prediction")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract and convert CSV data to TOML format
    Extract {
        #[arg(long, help = "Stock symbol (e.g., WEGE3)")]
        asset: String,
        #[arg(long, help = "Data source (e.g., investing)")]
        source: String,
        #[arg(long, default_value = "../dados", help = "Data directory path")]
        data_dir: String,
    },
    /// Train LSTM model on processed data
    Train {
        #[arg(long, help = "Stock symbol")]
        asset: String,
        #[arg(long, help = "Data source")]
        source: String,
        #[arg(long, default_value = "../dados", help = "Data directory path")]
        data_dir: String,
        #[arg(long, default_value_t = 10, help = "Sequence length for LSTM")]
        seq_length: usize,
        #[arg(long, default_value_t = 0.01, help = "Learning rate")]
        learning_rate: f32,
        #[arg(long, default_value_t = 50, help = "Number of training epochs")]
        epochs: usize,
        #[arg(long, default_value_t = 20, help = "LSTM hidden size")]
        hidden_size: usize,
        #[arg(long, default_value_t = 0.2, help = "Dropout rate for regularization")]
        dropout_rate: f32,
        #[arg(long, default_value_t = 0.8, help = "Train/validation split ratio")]
        train_split: f32,
    },
    /// Generate predictions using trained model
    Predict {
        #[arg(long, help = "Stock symbol")]
        asset: String,
        #[arg(long, help = "Data source")]
        source: String,
        #[arg(long, default_value = "../dados", help = "Data directory path")]
        data_dir: String,
        #[arg(long, default_value_t = 10, help = "Sequence length")]
        seq_length: usize,
        #[arg(long, default_value_t = 20, help = "Number of predictions to generate")]
        num_predictions: usize,
    },
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
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
    records: Vec<StockRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    asset: String,
    source: String,
    seq_length: usize,
    hidden_size: usize,
    train_samples: usize,
    validation_samples: usize,
    final_train_loss: f32,
    final_val_loss: f32,
    directional_accuracy: f32,
    mean: f32,
    std: f32,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedModel {
    metadata: ModelMetadata,
    lstm: LSTMCell,
    loss_history: Vec<(f32, f32)>, // (train_loss, val_loss)
}

#[derive(Debug, Serialize, Deserialize)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Vec<Vec<f32>>,
    weight_hh: Vec<Vec<f32>>,
    bias: Vec<f32>,
    weight_out: Vec<f32>,
    bias_out: f32,
}

#[derive(Debug, Serialize)]
struct PredictionResults {
    metadata: ModelMetadata,
    predictions: Vec<PredictionPoint>,
}

#[derive(Debug, Serialize)]
struct PredictionPoint {
    date: String,
    predicted_price: f32,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn parse_float(s: &str) -> Result<f32, LSTMError> {
    let cleaned = s.replace(',', ".").replace(" ", "").trim().to_string();
    cleaned.parse::<f32>()
        .map_err(|_| LSTMError::ParseError(format!("Invalid number: {}", s)))
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let binding = s.replace(',', ".").replace(" ", "");
    let cleaned = binding.trim();
    let multiplier = if cleaned.ends_with('M') || cleaned.ends_with('m') {
        1e6
    } else if cleaned.ends_with('K') || cleaned.ends_with('k') {
        1e3
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
    let value = cleaned.parse::<f32>()
        .map_err(|_| LSTMError::ParseError(format!("Invalid percentage: {}", s)))?;
    Ok(value / 100.0)
}

fn xavier_uniform(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> f32 {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    rng.gen_range(-limit..limit)
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v.clamp(-500.0, 500.0)).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.clamp(-500.0, 500.0).tanh())
}

// ============================================================================
// LSTM IMPLEMENTATION
// ============================================================================

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> LSTMCell {
        let mut rng = rand::thread_rng();
        
        let weight_ih = (0..4 * hidden_size)
            .map(|_| (0..input_size)
                .map(|_| xavier_uniform(input_size, 4 * hidden_size, &mut rng))
                .collect())
            .collect();
        
        let weight_hh = (0..4 * hidden_size)
            .map(|_| (0..hidden_size)
                .map(|_| xavier_uniform(hidden_size, 4 * hidden_size, &mut rng))
                .collect())
            .collect();
        
        let bias = (0..4 * hidden_size).map(|_| 0.0).collect();
        let weight_out = (0..hidden_size)
            .map(|_| xavier_uniform(hidden_size, 1, &mut rng))
            .collect();
        let bias_out = 0.0;
        
        LSTMCell {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias,
            weight_out,
            bias_out,
        }
    }

    fn forward(
        &self,
        x: &Array1<f32>,
        h_prev: &Array1<f32>,
        c_prev: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, f32) {
        let gates = {
            let ih = Array2::from_shape_vec(
                (4 * self.hidden_size, self.input_size),
                self.weight_ih.iter().flatten().cloned().collect(),
            ).unwrap();
            let hh = Array2::from_shape_vec(
                (4 * self.hidden_size, self.hidden_size),
                self.weight_hh.iter().flatten().cloned().collect(),
            ).unwrap();
            let bias = Array1::from_vec(self.bias.clone());
            &ih.dot(x) + &hh.dot(h_prev) + &bias
        };

        let i = sigmoid(&gates.slice(s![..self.hidden_size]).to_owned());
        let f = sigmoid(&gates.slice(s![self.hidden_size..2 * self.hidden_size]).to_owned());
        let o = sigmoid(&gates.slice(s![2 * self.hidden_size..3 * self.hidden_size]).to_owned());
        let g = tanh(&gates.slice(s![3 * self.hidden_size..]).to_owned());

        let c = &f * c_prev + &i * &g;
        let h = &o * tanh(&c);

        let output = h.dot(&Array1::from_vec(self.weight_out.clone())) + self.bias_out;
        
        (h.to_owned(), c.to_owned(), output)
    }

    fn update_weights(
        &mut self,
        x: &Array1<f32>,
        h_prev: &Array1<f32>,
        c_prev: &Array1<f32>,
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let (h, _c, pred) = self.forward(x, h_prev, c_prev);
        let loss = (pred - target).powi(2);

        let d_loss_d_pred = 2.0 * (pred - target);
        let d_pred_d_wo = h.clone();
        let d_pred_d_bo = 1.0;

        for (wo, &d_wo) in self.weight_out.iter_mut().zip(d_pred_d_wo.iter()) {
            *wo -= learning_rate * d_loss_d_pred * d_wo;
        }
        self.bias_out -= learning_rate * d_loss_d_pred * d_pred_d_bo;

        let d_pred_d_h = Array1::from_vec(self.weight_out.clone());
        let d_h = d_loss_d_pred * &d_pred_d_h;

        for (i, wi) in self.weight_ih.iter_mut().enumerate() {
            if i < self.hidden_size {
                for (j, w) in wi.iter_mut().enumerate() {
                    if j < x.len() {
                        *w -= learning_rate * d_h[i % self.hidden_size] * x[j] * 0.1;
                    }
                }
            }
        }

        for (i, wh) in self.weight_hh.iter_mut().enumerate() {
            if i < self.hidden_size {
                for (j, w) in wh.iter_mut().enumerate() {
                    if j < h_prev.len() {
                        *w -= learning_rate * d_h[i % self.hidden_size] * h_prev[j] * 0.1;
                    }
                }
            }
        }

        loss
    }

    fn predict(&self, sequence: &[f32], mean: f32, std: f32) -> f32 {
        let mut h_t = Array1::zeros(self.hidden_size);
        let mut c_t = Array1::zeros(self.hidden_size);
        let mut output = 0.0;

        for &val in sequence {
            let x = Array1::from_vec(vec![val]);
            let (h_next, c_next, pred) = self.forward(&x, &h_t, &c_t);
            h_t = h_next;
            c_t = c_next;
            output = pred;
        }

        output * std + mean
    }
}

// ============================================================================
// COMMAND IMPLEMENTATIONS
// ============================================================================

fn extract_command(asset: String, source: String, data_dir: String) -> Result<(), LSTMError> {
    info!("Starting data extraction for {} from {}", asset, source);
    
    let input_file_path = format!("{}/{}/{}.csv", data_dir, source, asset);
    let output_file_path = format!("{}/{}_{}_output.toml", data_dir, asset, source);

    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        return Err(LSTMError::FileNotFound { path: input_file_path });
    }

    info!("Reading CSV file: {}", input_file_path);
    let file = File::open(&input_file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();
    for (line_num, result) in rdr.records().enumerate() {
        match result {
            Ok(record) => {
                if record.len() >= 7 {
                    match (|| -> Result<StockRecord, LSTMError> {
                        let date = record[0].to_string();
                        let closing = parse_float(&record[1])?;
                        let opening = parse_float(&record[2])?;
                        let high = parse_float(&record[3])?;
                        let low = parse_float(&record[4])?;
                        let volume = parse_volume(&record[5])?;
                        let variation = parse_percentage(&record[6])?;

                        if closing <= 0.0 || opening <= 0.0 || high <= 0.0 || low <= 0.0 {
                            return Err(LSTMError::ParseError("Prices must be positive".into()));
                        }
                        if high < low {
                            return Err(LSTMError::ParseError("High price cannot be less than low price".into()));
                        }

                        Ok(StockRecord { date, closing, opening, high, low, volume, variation })
                    })() {
                        Ok(stock_record) => records.push(stock_record),
                        Err(e) => {
                            warn!("Skipping invalid record at line {}: {}", line_num + 1, e);
                            continue;
                        }
                    }
                } else {
                    warn!("Skipping record with insufficient columns at line {}: {:?}", line_num + 1, record);
                }
            }
            Err(e) => {
                warn!("Error reading record at line {}: {}", line_num + 1, e);
            }
        }
    }

    if records.is_empty() {
        return Err(LSTMError::InvalidCsv { msg: "No valid records found".into() });
    }

    info!("Successfully parsed {} records", records.len());

    let stock_data = StockData { records };
    let toml_data = toml::to_string_pretty(&stock_data)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;

    let output_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_file_path)?;

    std::io::Write::write_all(&mut std::io::BufWriter::new(output_file), toml_data.as_bytes())?;

    info!("Data successfully saved to: {}", output_file_path);
    println!("✅ Extraction complete! {} records saved to {}", stock_data.records.len(), output_file_path);

    Ok(())
}

fn train_command(
    asset: String,
    source: String,
    data_dir: String,
    seq_length: usize,
    learning_rate: f32,
    epochs: usize,
    hidden_size: usize,
    _dropout_rate: f32,
    train_split: f32,
) -> Result<(), LSTMError> {
    info!("Starting LSTM training for {} from {}", asset, source);
    
    let input_file_path = format!("{}/{}_{}_output.toml", data_dir, asset, source);
    let output_model_path = format!("{}/{}_{}_lstm_model.json", data_dir, asset, source);

    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        return Err(LSTMError::FileNotFound { path: input_file_path });
    }

    info!("Loading data from: {}", input_file_path);
    let toml_data = std::fs::read_to_string(&input_file_path)?;
    let stock_data: StockData = toml::from_str(&toml_data)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;
    let records = stock_data.records;

    if records.len() <= seq_length + 1 {
        return Err(LSTMError::InsufficientData {
            required: seq_length + 2,
            actual: records.len(),
        });
    }

    let prices: Array1<f32> = Array1::from_vec(records.iter().map(|r| r.closing).collect());
    let mean = prices.mean().unwrap_or(0.0);
    let std = prices.std_axis(ndarray::Axis(0), 0.0).into_scalar();
    
    if std == 0.0 {
        return Err(LSTMError::ParseError("Standard deviation is zero - no price variation".into()));
    }

    let normalized_prices = prices.mapv(|x| (x - mean) / std);

    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    for i in 0..(normalized_prices.len() - seq_length) {
        let seq = normalized_prices.slice(s![i..i + seq_length]).to_vec();
        let target = normalized_prices[i + seq_length];
        sequences.push(seq);
        targets.push(target);
    }

    let split_idx = (train_split * sequences.len() as f32) as usize;
    let (train_seqs, val_seqs) = sequences.split_at(split_idx);
    let (train_targets, val_targets) = targets.split_at(split_idx);

    info!("Training samples: {}, Validation samples: {}", train_seqs.len(), val_seqs.len());
    println!("🧠 Training LSTM: {} train samples, {} validation samples", train_seqs.len(), val_seqs.len());

    let mut lstm = LSTMCell::new(1, hidden_size);
    let mut loss_history = Vec::new();
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;
    let patience = 10;
    let mut directional_accuracy = 0.0;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        for i in 0..train_seqs.len() {
            let mut h_t = Array1::zeros(hidden_size);
            let mut c_t = Array1::zeros(hidden_size);
            
            for &input_val in &train_seqs[i] {
                let x = Array1::from_vec(vec![input_val]);
                let (h_next, c_next, _output) = lstm.forward(&x, &h_t, &c_t);
                h_t = h_next;
                c_t = c_next;
            }
            
            let last_x = Array1::from_vec(vec![*train_seqs[i].last().unwrap()]);
            let loss = lstm.update_weights(&last_x, &h_t, &c_t, train_targets[i], learning_rate);
            total_loss += loss;
        }
        let avg_train_loss = total_loss / train_seqs.len() as f32;

        let mut val_loss = 0.0;
        let mut correct_direction = 0;
        for i in 0..val_seqs.len() {
            let mut h_t = Array1::zeros(hidden_size);
            let mut c_t = Array1::zeros(hidden_size);
            
            for &input_val in &val_seqs[i] {
                let x = Array1::from_vec(vec![input_val]);
                let (h_next, c_next, _output) = lstm.forward(&x, &h_t, &c_t);
                h_t = h_next;
                c_t = c_next;
            }
            
            let last_x = Array1::from_vec(vec![*val_seqs[i].last().unwrap()]);
            let (_, _, pred) = lstm.forward(&last_x, &h_t, &c_t);
            let loss = (pred - val_targets[i]).powi(2);
            val_loss += loss;

            if val_seqs[i].len() > 1 {
                let last_actual = *val_seqs[i].last().unwrap();
                let prev_actual = val_seqs[i][val_seqs[i].len() - 2];
                let actual_direction = val_targets[i] > last_actual;
                let pred_direction = pred > last_actual;
                if actual_direction == pred_direction {
                    correct_direction += 1;
                }
            }
        }
        let avg_val_loss = val_loss / val_seqs.len() as f32;
        directional_accuracy = correct_direction as f32 / val_seqs.len() as f32;

        loss_history.push((avg_train_loss, avg_val_loss));

        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!("Epoch {}/{}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.2}%", 
                  epoch + 1, epochs, avg_train_loss, avg_val_loss, directional_accuracy * 100.0);
            println!("📊 Epoch {}/{}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.2}%", 
                     epoch + 1, epochs, avg_train_loss, avg_val_loss, directional_accuracy * 100.0);
        }

        if patience_counter >= patience {
            info!("Early stopping triggered at epoch {}", epoch + 1);
            println!("⏹️  Early stopping - validation loss stopped improving");
            break;
        }
    }

    let metadata = ModelMetadata {
        asset: asset.clone(),
        source: source.clone(),
        seq_length,
        hidden_size,
        train_samples: train_seqs.len(),
        validation_samples: val_seqs.len(),
        final_train_loss: loss_history.last().map(|x| x.0).unwrap_or(0.0),
        final_val_loss: loss_history.last().map(|x| x.1).unwrap_or(0.0),
        directional_accuracy,
        mean,
        std,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    let saved_model = SavedModel {
        metadata,
        lstm,
        loss_history,
    };

    let model_json = serde_json::to_string_pretty(&saved_model)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;
    std::fs::write(&output_model_path, model_json)?;

    info!("Model saved to: {}", output_model_path);
    println!("✅ Training complete! Model saved to {}", output_model_path);

    Ok(())
}

fn predict_command(
    asset: String,
    source: String,
    data_dir: String,
    seq_length: usize,
    num_predictions: usize,
) -> Result<(), LSTMError> {
    info!("Starting prediction for {} from {}", asset, source);
    
    let data_path = format!("{}/{}_{}_output.toml", data_dir, asset, source);
    let model_path = format!("{}/{}_{}_lstm_model.json", data_dir, asset, source);
    let output_path = format!("{}/{}_{}_predictions.json", data_dir, asset, source);

    // Load model
    if !Path::new(&model_path).exists() {
        return Err(LSTMError::FileNotFound { path: model_path });
    }
    
    info!("Loading model from: {}", model_path);
    let model_json = std::fs::read_to_string(&model_path)?;
    let saved_model: SavedModel = serde_json::from_str(&model_json)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;

    // Load data
    if !Path::new(&data_path).exists() {
        return Err(LSTMError::FileNotFound { path: data_path });
    }
    
    info!("Loading data from: {}", data_path);
    let toml_data = std::fs::read_to_string(&data_path)?;
    let stock_data: StockData = toml::from_str(&toml_data)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;
    let records = stock_data.records;

    if records.len() < seq_length {
        return Err(LSTMError::InsufficientData {
            required: seq_length,
            actual: records.len(),
        });
    }

    // Get the last sequence of data for initialization
    let mut sequence: Vec<f32> = records
        .iter()
        .rev()
        .take(seq_length)
        .map(|r| (r.closing - saved_model.metadata.mean) / saved_model.metadata.std)
        .collect();
    sequence.reverse(); // Ensure correct temporal order (oldest to newest)

    // Parse the last date in the dataset
    let last_date_str = records.last().unwrap().date.clone();
    let last_date = NaiveDate::parse_from_str(&last_date_str, "%d.%m.%Y")
        .map_err(|e| LSTMError::ParseError(format!("Invalid date format: {}", e)))?;
    
    // Generate future predictions
    let mut predictions = Vec::new();
    let mut h_t = Array1::zeros(saved_model.metadata.hidden_size);
    let mut c_t = Array1::zeros(saved_model.metadata.hidden_size);
    
    println!("🔮 Generating {} future predictions starting from 29.07.2025...", num_predictions);
    
    for i in 0..num_predictions {
        // Predict the next price
        let pred_normalized = {
            let mut temp_h_t = h_t.clone();
            let mut temp_c_t = c_t.clone();
            let mut output = 0.0;
            for &val in &sequence {
                let x = Array1::from_vec(vec![val]);
                let (h_next, c_next, pred) = saved_model.lstm.forward(&x, &temp_h_t, &temp_c_t);
                temp_h_t = h_next;
                temp_c_t = c_next;
                output = pred;
            }
            h_t = temp_h_t;
            c_t = temp_c_t;
            output
        };
        let pred = pred_normalized * saved_model.metadata.std + saved_model.metadata.mean; // Denormalize
        
        // Generate the next date (add i+1 days to last date)
        let next_date = last_date + Duration::days((i + 1) as i64);
        let date_str = next_date.format("%d.%m.%Y").to_string();
        
        predictions.push(PredictionPoint {
            date: date_str.clone(),
            predicted_price: pred,
        });
        
        println!("📈 {}: Predicted = R$ {:.2}", date_str, pred);
        
        // Update sequence for the next prediction
        sequence.remove(0); // Remove oldest value
        sequence.push(pred_normalized); // Add new prediction (normalized)
    }

    // Create and save results
    let results = PredictionResults {
        metadata: saved_model.metadata,
        predictions,
    };

    let results_json = serde_json::to_string_pretty(&results)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;
    std::fs::write(&output_path, results_json)?;

    info!("Predictions saved to: {}", output_path);
    println!("✅ Predictions saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Extract { asset, source, data_dir } => {
            extract_command(asset, source, data_dir)
        }
        Commands::Train { 
            asset, 
            source, 
            data_dir, 
            seq_length, 
            learning_rate, 
            epochs, 
            hidden_size, 
            dropout_rate, 
            train_split 
        } => {
            train_command(
                asset, 
                source, 
                data_dir, 
                seq_length, 
                learning_rate, 
                epochs, 
                hidden_size, 
                dropout_rate, 
                train_split
            )
        }
        Commands::Predict { asset, source, data_dir, seq_length, num_predictions } => {
            predict_command(asset, source, data_dir, seq_length, num_predictions)
        }
    };

    match result {
        Ok(()) => {
            info!("Command completed successfully");
            Ok(())
        }
        Err(e) => {
            error!("Command failed: {}", e);
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}


// cd mlpmercadofinanc

 
// # 1. Extrair dados do CSV
// cargo run -- extract --asset WEGE3 --source investing

//# 2. Treinar o modelo LSTM
// cargo run -- train --asset WEGE3 --source investing --epochs 100 --learning-rate 0.01 --hidden-size 20

//# 3. Gerar predições
// cargo run -- predict --asset WEGE3 --source investing --num-predictions 20



// # Experimente diferentes hiperparâmetros:
// cargo run -- train --asset WEGE3 --source investing --epochs 200 --learning-rate 0.005 --hidden-size 32

// # Para sequências maiores:
// cargo run -- train --asset WEGE3 --source investing --seq-length 20 --hidden-size 50



// cargo run -- train --asset WEGE3 --source investing --epochs 100 --learning-rate 0.01 --hidden-size 20 --seq-length 60