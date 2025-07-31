// projeto : lstmfiletrain
// file : src/main.rs
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json;
use ndarray::{Array1, Array2, s, Axis};
use log::info;
use env_logger;
use thiserror::Error;
use chrono::{Utc};
use rusqlite::{Connection, params};
use postgres::{Client, NoTls};
use rand::distributions::{Distribution, Uniform};

#[derive(Error, Debug)]
enum LSTMError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Insufficient data: need at least {required} records, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
}

#[derive(Parser)]
#[command(name = "lstm_train")]
#[command(about = "Train LSTM model on processed data")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(long, help = "Stock symbol")]
    asset: String,
    #[arg(long, help = "Data source")]
    source: String,
    #[arg(long, default_value = "../dados", help = "Data directory path")]
    data_dir: String,
    #[arg(long, default_value_t = 20, help = "Sequence length for LSTM")]
    seq_length: usize,
    #[arg(long, default_value_t = 0.001, help = "Learning rate")]
    learning_rate: f32,
    #[arg(long, default_value_t = 5, help = "Number of training epochs")]
    epochs: usize,
    #[arg(long, default_value_t = 50, help = "LSTM hidden size")]
    hidden_size: usize,
    #[arg(long, default_value_t = 0.8, help = "Train/validation split ratio")]
    train_split: f32,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db", help = "PostgreSQL connection string")]
    pg_conn_string: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Vec<Vec<f32>>,
    weight_hh: Vec<Vec<f32>>,
    bias: Vec<f32>,
    weight_out: Vec<f32>,
    bias_out: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedModel {
    metadata: ModelMetadata,
    lstm: LSTMCell,
    loss_history: Vec<(f32, f32)>,
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v.clamp(-80.0, 80.0)).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.clamp(-80.0, 80.0).tanh())
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let uniform = Uniform::new(-0.1, 0.1); // Xavier initialization
        let mut rng = rand::thread_rng();
        
        let weight_ih = (0..4 * hidden_size)
            .map(|_| (0..input_size)
                .map(|_| uniform.sample(&mut rng))
                .collect())
            .collect();
        let weight_hh = (0..4 * hidden_size)
            .map(|_| (0..hidden_size)
                .map(|_| uniform.sample(&mut rng))
                .collect())
            .collect();
        let mut bias = vec![0.0; 4 * hidden_size];
        // Initialize forget gate biases to 1.0
        for i in hidden_size..2 * hidden_size {
            bias[i] = 1.0;
        }
        let weight_out = (0..hidden_size)
            .map(|_| uniform.sample(&mut rng))
            .collect();
        let bias_out = 0.0;

        Self {
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
    ) -> (Array1<f32>, Array1<f32>, f32, Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
        let ih = Array2::from_shape_vec(
            (4 * self.hidden_size, self.input_size),
            self.weight_ih.iter().flatten().cloned().collect(),
        ).expect("Failed to reshape weight_ih");
        let hh = Array2::from_shape_vec(
            (4 * self.hidden_size, self.hidden_size),
            self.weight_hh.iter().flatten().cloned().collect(),
        ).expect("Failed to reshape weight_hh");
        let bias = Array1::from_vec(self.bias.clone());

        let gates = &ih.dot(x) + &hh.dot(h_prev) + &bias;
        let hidden_size = self.hidden_size;

        let i = sigmoid(&gates.slice(s![0..hidden_size]).to_owned());
        let f = sigmoid(&gates.slice(s![hidden_size..2 * hidden_size]).to_owned());
        let o = sigmoid(&gates.slice(s![2 * hidden_size..3 * hidden_size]).to_owned());
        let g = tanh(&gates.slice(s![3 * hidden_size..4 * hidden_size]).to_owned());

        let c = &f * c_prev + &i * &g;
        let h = &o * tanh(&c);

        let output = h.dot(&Array1::from_vec(self.weight_out.clone())) + self.bias_out;

        (h, c, output, i, f, o, g)
    }

    fn update_weights(
        &mut self,
        sequence: &[f32],
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let seq_len = sequence.len();
        if seq_len == 0 {
            return 0.0;
        }

        let mut h_states = vec![Array1::zeros(self.hidden_size); seq_len + 1];
        let mut c_states = vec![Array1::zeros(self.hidden_size); seq_len + 1];
        let mut outputs = vec![0.0; seq_len];
        let mut i_gates = vec![Array1::zeros(self.hidden_size); seq_len];
        let mut f_gates = vec![Array1::zeros(self.hidden_size); seq_len];
        let mut o_gates = vec![Array1::zeros(self.hidden_size); seq_len];
        let mut g_gates = vec![Array1::zeros(self.hidden_size); seq_len];

        // Forward pass
        for t in 0..seq_len {
            let x = Array1::from_vec(vec![sequence[t]]);
            let (h_next, c_next, output, i, f, o, g) = self.forward(&x, &h_states[t], &c_states[t]);
            h_states[t + 1] = h_next;
            c_states[t + 1] = c_next;
            outputs[t] = output;
            i_gates[t] = i;
            f_gates[t] = f;
            o_gates[t] = o;
            g_gates[t] = g;
        }

        let final_output = outputs[seq_len - 1];
        let loss = (final_output - target).powi(2);
        let d_loss_d_output = 2.0 * (final_output - target);

        // Backward pass (BPTT)
        let mut d_weight_out = vec![0.0; self.hidden_size];
        let mut d_bias_out = 0.0;
        let mut d_weight_ih = vec![vec![0.0; self.input_size]; 4 * self.hidden_size];
        let mut d_weight_hh = vec![vec![0.0; self.hidden_size]; 4 * self.hidden_size];
        let mut d_bias = vec![0.0; 4 * self.hidden_size];

        let mut d_h_next = Array1::from_vec(vec![d_loss_d_output; self.hidden_size]);
        let mut d_c_next = Array1::zeros(self.hidden_size);

        for t in (0..seq_len).rev() {
            let h_t = &h_states[t + 1];
            let c_t = &c_states[t + 1];
            let c_prev = &c_states[t];
            let x = Array1::from_vec(vec![sequence[t]]);

            let d_o = &d_h_next * &tanh(&c_t) * &o_gates[t] * &(1.0 - &o_gates[t]);
            let d_c = &d_h_next * &o_gates[t] * &(1.0 - &tanh(&c_t).mapv(|v| v.powi(2))) + &d_c_next;
            let d_f = &d_c * c_prev * &f_gates[t] * &(1.0 - &f_gates[t]);
            let d_i = &d_c * &g_gates[t] * &i_gates[t] * &(1.0 - &i_gates[t]);
            let d_g = &d_c * &i_gates[t] * &(1.0 - &g_gates[t].mapv(|v| v.powi(2)));

            let d_gates = {
                let mut gates = Array1::zeros(4 * self.hidden_size);
                gates.slice_mut(s![0..self.hidden_size]).assign(&d_i);
                gates.slice_mut(s![self.hidden_size..2 * self.hidden_size]).assign(&d_f);
                gates.slice_mut(s![2 * self.hidden_size..3 * self.hidden_size]).assign(&d_o);
                gates.slice_mut(s![3 * self.hidden_size..4 * self.hidden_size]).assign(&d_g);
                gates
            };

            let ih = Array2::from_shape_vec(
                (4 * self.hidden_size, self.input_size),
                self.weight_ih.iter().flatten().cloned().collect(),
            ).expect("Failed to reshape weight_ih");
            let hh = Array2::from_shape_vec(
                (4 * self.hidden_size, self.hidden_size),
                self.weight_hh.iter().flatten().cloned().collect(),
            ).expect("Failed to reshape weight_hh");

            // Update gradients
            for i in 0..self.hidden_size {
                d_weight_out[i] += d_loss_d_output * h_t[i];
                d_bias_out += d_loss_d_output;
            }

            for i in 0..4 * self.hidden_size {
                for j in 0..self.input_size {
                    d_weight_ih[i][j] += d_gates[i] * x[j];
                }
                for j in 0..self.hidden_size {
                    d_weight_hh[i][j] += d_gates[i] * h_states[t][j];
                }
                d_bias[i] += d_gates[i];
            }

            // Gradient clipping
            let clip_value = 1.0;
            for i in 0..4 * self.hidden_size {
                for j in 0..self.input_size {
                    d_weight_ih[i][j] = d_weight_ih[i][j].clamp(-clip_value, clip_value);
                }
                for j in 0..self.hidden_size {
                    d_weight_hh[i][j] = d_weight_hh[i][j].clamp(-clip_value, clip_value);
                }
                d_bias[i] = d_bias[i].clamp(-clip_value, clip_value);
            }
            for i in 0..self.hidden_size {
                d_weight_out[i] = d_weight_out[i].clamp(-clip_value, clip_value);
            }
            d_bias_out = d_bias_out.clamp(-clip_value, clip_value);

            // Update weights
            for i in 0..4 * self.hidden_size {
                for j in 0..self.input_size {
                    self.weight_ih[i][j] -= learning_rate * d_weight_ih[i][j];
                }
                for j in 0..self.hidden_size {
                    self.weight_hh[i][j] -= learning_rate * d_weight_hh[i][j];
                }
                self.bias[i] -= learning_rate * d_bias[i];
            }
            for i in 0..self.hidden_size {
                self.weight_out[i] -= learning_rate * d_weight_out[i];
            }
            self.bias_out -= learning_rate * d_bias_out;

            // Propagate gradients backward
            d_h_next = hh.t().dot(&d_gates);
            d_c_next = &d_c * &f_gates[t];
        }

        loss
    }
}

fn train_command(
    asset: String,
    source: String,
    data_dir: String,
    seq_length: usize,
    mut learning_rate: f32,
    epochs: usize,
    hidden_size: usize,
    train_split: f32,
    pg_conn_string: String,
) -> Result<(), LSTMError> {
    info!("Starting LSTM training for {} from {}", asset, source);
    let db_file_path = format!("{}/{}.db", data_dir, source);
    let conn = Connection::open(&db_file_path)?;
    let mut stmt = conn.prepare("SELECT date, closing, opening, high, low, volume, variation FROM stock_records WHERE asset = ?1 ORDER BY date")?;
    let records_iter = stmt.query_map(params![&asset], |row| {
        Ok(StockRecord {
            date: row.get(0)?,
            closing: row.get(1)?,
            opening: row.get(2)?,
            high: row.get(3)?,
            low: row.get(4)?,
            volume: row.get(5)?,
            variation: row.get(6)?,
        })
    })?;

    let mut records = Vec::new();
    for record_result in records_iter {
        records.push(record_result?);
    }

    if records.is_empty() {
        return Err(LSTMError::FileNotFound { path: format!("No records found for asset {} in {}", asset, db_file_path) });
    }
    if records.len() <= seq_length {
        return Err(LSTMError::InsufficientData {
            required: seq_length + 1,
            actual: records.len(),
        });
    }

    // Log data for debugging
    info!("Loaded {} records", records.len());
    println!("📈 Loaded {} records", records.len());
    for record in records.iter().take(5).chain(records.iter().rev().take(5)) {
        println!("Date: {}, Closing: {:.2}", record.date, record.closing);
    }

    let prices: Array1<f32> = Array1::from_vec(records.iter().map(|r| r.closing).collect());
    let mean = prices.mean().unwrap_or(0.0);
    let std = prices.std_axis(Axis(0), 0.0).into_scalar();
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
    if split_idx >= sequences.len() || split_idx == 0 {
        return Err(LSTMError::InsufficientData {
            required: 2,
            actual: sequences.len(),
        });
    }
    let (train_seqs, val_seqs) = sequences.split_at(split_idx);
    let (train_targets, val_targets) = targets.split_at(split_idx);

    info!("Training samples: {}, Validation samples: {}", train_seqs.len(), val_seqs.len());
    println!("🧠 Training LSTM: {} train samples, {} validation samples", train_seqs.len(), val_seqs.len());

    let mut lstm = LSTMCell::new(1, hidden_size);
    let mut loss_history = Vec::new();
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;
    const PATIENCE: usize = 20;
    let lr_decay = 0.95;

    for epoch in 0..epochs {
        let mut total_train_loss = 0.0;
        for i in 0..train_seqs.len() {
            let loss = lstm.update_weights(&train_seqs[i], train_targets[i], learning_rate);
            total_train_loss += loss;
        }
        let avg_train_loss = total_train_loss / train_seqs.len() as f32;

        let mut total_val_loss = 0.0;
        let mut correct_direction = 0;
        let mut total_direction_checks = 0;
        for i in 0..val_seqs.len() {
            let mut h_t = Array1::zeros(hidden_size);
            let mut c_t = Array1::zeros(hidden_size);
            let mut final_output = 0.0;
            for &input_val in &val_seqs[i] {
                let x = Array1::from_vec(vec![input_val]);
                let (h_next, c_next, output, _, _, _, _) = lstm.forward(&x, &h_t, &c_t);
                h_t = h_next;
                c_t = c_next;
                final_output = output;
            }
            let loss = (final_output - val_targets[i]).powi(2);
            total_val_loss += loss;

            if val_seqs[i].len() > 1 {
                let last_input = *val_seqs[i].last().unwrap();
                let actual_direction = val_targets[i] > last_input;
                let pred_direction = final_output > last_input;
                if actual_direction == pred_direction {
                    correct_direction += 1;
                }
                total_direction_checks += 1;
            }
        }
        let avg_val_loss = total_val_loss / val_seqs.len() as f32;
        let directional_accuracy = if total_direction_checks > 0 {
            correct_direction as f32 / total_direction_checks as f32
        } else {
            0.0
        };

        loss_history.push((avg_train_loss, avg_val_loss));

        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        learning_rate *= lr_decay;

        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!(
                "Epoch {}/{}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.2}%, LR: {:.6}",
                epoch + 1,
                epochs,
                avg_train_loss,
                avg_val_loss,
                directional_accuracy * 100.0,
                learning_rate
            );
            println!(
                "📊 Epoch {}/{}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.2}%, LR: {:.6}",
                epoch + 1,
                epochs,
                avg_train_loss,
                avg_val_loss,
                directional_accuracy * 100.0,
                learning_rate
            );
        }

        if patience_counter >= PATIENCE {
            info!("Early stopping triggered at epoch {}", epoch + 1);
            println!("⏹️ Early stopping - validation loss stopped improving");
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
        directional_accuracy: {
            let mut correct = 0;
            let mut total = 0;
            for i in 0..val_seqs.len() {
                let mut h_t = Array1::zeros(hidden_size);
                let mut c_t = Array1::zeros(hidden_size);
                let mut final_output = 0.0;
                for &input_val in &val_seqs[i] {
                    let x = Array1::from_vec(vec![input_val]);
                    let (h_next, c_next, output, _, _, _, _) = lstm.forward(&x, &h_t, &c_t);
                    h_t = h_next;
                    c_t = c_next;
                    final_output = output;
                }
                if val_seqs[i].len() > 1 {
                    let last_input = *val_seqs[i].last().unwrap();
                    let actual_direction = val_targets[i] > last_input;
                    let pred_direction = final_output > last_input;
                    if actual_direction == pred_direction {
                        correct += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 { correct as f32 / total as f32 } else { 0.0 }
        },
        mean,
        std,
        timestamp: Utc::now().to_rfc3339(),
    };

    let saved_model = SavedModel {
        metadata,
        lstm,
        loss_history,
    };

    let model_json = serde_json::to_string(&saved_model)
        .map_err(|e| LSTMError::SerializationError(e.to_string()))?;

    let mut pg_client = Client::connect(&pg_conn_string, NoTls)?;
    pg_client.execute(
        "CREATE TABLE IF NOT EXISTS lstm_models (
            asset TEXT NOT NULL,
            source TEXT NOT NULL,
            model_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (asset, source)
        )",
        &[],
    )?;
    pg_client.execute(
        "DELETE FROM lstm_models WHERE asset = $1 AND source = $2",
        &[&asset, &source],
    )?;
    pg_client.execute(
        "INSERT INTO lstm_models (asset, source, model_json, timestamp) VALUES ($1, $2, $3, $4)",
        &[&asset, &source, &model_json, &Utc::now().to_rfc3339()],
    )?;

    info!("Model saved to PostgreSQL database");
    println!("✅ Training complete! Model saved to PostgreSQL database");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    let cli = Cli::parse();
    let result = train_command(
        cli.asset,
        cli.source,
        cli.data_dir,
        cli.seq_length,
        cli.learning_rate,
        cli.epochs,
        cli.hidden_size,
        cli.train_split,
        cli.pg_conn_string,
    );
    match result {
        Ok(()) => {
            info!("Command completed successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    }
}
 
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50 --learning-rate 0.001 --epochs 100



// cd lstmfiletrain
// cargo run -- --asset WEGE3 --source investing

// # 2. Train the LSTM model
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*

// "Data","Último","Abertura","Máxima","Mínima","Vol.","Var%"
// "29.07.2025","37,53","37,15","37,94","36,81","6,10M","2,18%"