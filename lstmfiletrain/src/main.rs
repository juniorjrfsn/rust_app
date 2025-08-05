// projeto: lstmfiletrain
// file: src/main.rs

 
 
mod rna {
    pub mod model;
    pub mod data;
    pub mod storage;
    pub mod metrics;
}

use clap::Parser;
use log::{info, error};
use env_logger;
use thiserror::Error;
use chrono::Utc;
use postgres::{Client, NoTls};
use std::time::Instant;
use rna::model::MultiLayerLSTM;
use rna::data::{load_all_assets, load_data_from_postgres, create_sequences, normalize_data};
use rna::storage::save_model_to_postgres;
use rna::metrics::{TrainingMetrics, calculate_rmse, calculate_mape, calculate_directional_accuracy};

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Insufficient data for asset '{asset}': need at least {required}, got {actual}")]
    InsufficientData { asset: String, required: usize, actual: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("PostgreSQL error: {0}")]
    PgError(#[from] postgres::Error),
    #[error("Training error: {0}")]
    TrainingError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Data loading error: {0}")]
    DataLoadingError(String),
    #[error("Date parse error: {0}")]
    DateParseError(String),
}

#[derive(Parser)]
#[command(name = "lstm_train", about = "Enhanced LSTM model for stock price prediction using PostgreSQL data", version = "3.0.1")]
struct Cli {
    #[arg(long, default_value = "../dados", help = "Data directory (compatibility only)")]
    data_dir: String,
    #[arg(long, default_value_t = 40, help = "Sequence length")]
    seq_length: usize,
    #[arg(long, default_value_t = 0.0005, help = "Initial learning rate")]
    learning_rate: f32,
    #[arg(long, default_value_t = 50, help = "Training epochs")]
    epochs: usize,
    #[arg(long, default_value_t = 64, help = "Hidden layer size")]
    hidden_size: usize,
    #[arg(long, default_value_t = 0.8, help = "Train/validation split")]
    train_split: f32,
    #[arg(long, default_value_t = 16, help = "Batch size")]
    batch_size: usize,
    #[arg(long, default_value_t = 0.3, help = "Dropout rate")]
    dropout_rate: f32,
    #[arg(long, default_value_t = 2, help = "Number of LSTM layers")]
    num_layers: usize,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
}

fn train_model(cli: Cli) -> Result<(), LSTMError> {
    let total_start_time = Instant::now();
    println!("🚀 Starting training for all assets (Enhanced Version)");

    let mut pg_client = Client::connect(&cli.pg_conn, NoTls)?;
    let assets = load_all_assets(&mut pg_client)?;
    println!("📊 Found {} unique assets to process", assets.len());

    let mut overall_metrics = Vec::new();

    for asset in assets {
        println!("\n🔍 Processing asset: {}", asset);
        let start_time = Instant::now();

        let data = load_data_from_postgres(&mut pg_client, &asset)?;
        if data.len() < cli.seq_length + 100 {
            println!("⚠️ Skipping asset {}: insufficient data ({} records, need {})", asset, data.len(), cli.seq_length + 100);
            continue;
        }
        println!("📊 Loaded {} records for asset {}", data.len(), asset);

        let (sequences, targets) = create_sequences(&data, cli.seq_length);
        let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
        let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
        let (_norm_closing, closing_mean, closing_std) = normalize_data(&closing_prices);
        let (_norm_opening, opening_mean, opening_std) = normalize_data(&opening_prices);

        info!("Data normalized for {} - Closing Mean: {:.4}, Closing Std: {:.4}, Opening Mean: {:.4}, Opening Std: {:.4}",
              asset, closing_mean, closing_std, opening_mean, opening_std);
        info!("Created {} sequences of length {} (5 features per timestep) for {}", sequences.len(), cli.seq_length, asset);

        let split_idx = (sequences.len() as f32 * cli.train_split) as usize;
        let (train_sequences, train_targets) = (sequences[..split_idx].to_vec(), targets[..split_idx].to_vec());
        let (val_sequences, val_targets) = (sequences[split_idx..].to_vec(), targets[split_idx..].to_vec());

        println!("🧠 Training: {} samples, Validation: {} samples for {}", train_sequences.len(), val_sequences.len(), asset);
        println!("🏗️ Model: {} layers, {} hidden units, {:.1}% dropout", cli.num_layers, cli.hidden_size, cli.dropout_rate * 100.0);

        let mut rng = rand::rng();
        let input_size = 5; // Closing, opening, MA5, momentum, RSI
        let mut model = MultiLayerLSTM::new(input_size, cli.hidden_size, cli.num_layers, cli.dropout_rate, &mut rng);

        let mut best_val_loss = f32::INFINITY;
        let mut patience = 0;
        const MAX_PATIENCE: usize = 20;
        let mut lr = cli.learning_rate;

        println!("🎯 Starting enhanced training for {}...", asset);

        for epoch in 0..cli.epochs {
            let train_batches = rna::data::create_batches(train_sequences.clone(), train_targets.clone(), cli.batch_size);
            let mut epoch_loss: f32 = 0.0;

            for (batch_seqs, batch_targets) in train_batches {
                if !batch_seqs.is_empty() && batch_seqs.len() == batch_targets.len() {
                    epoch_loss += model.train_step(&batch_seqs, &batch_targets, lr);
                }
            }
            epoch_loss /= (train_sequences.len() / cli.batch_size.max(1)) as f32;

            let mut val_loss = 0.0;
            let mut val_predictions = Vec::with_capacity(val_sequences.len());
            for (seq, &target) in val_sequences.iter().zip(val_targets.iter()) {
                let pred = model.forward(seq, false).min(3.0); // Cap normalized predictions
                val_predictions.push(pred);
                val_loss += (pred - target).powi(2);
            }
            val_loss /= val_sequences.len().max(1) as f32;

            let directional_acc = calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);

            if epoch > 0 && epoch % 30 == 0 {
                lr *= 0.9;
            }
            if epoch % 5 == 0 || epoch < 10 {
                println!("Epoch {:3}: Train Loss: {:.6}, Val Loss: {:.6}, Dir Acc: {:.1}%, LR: {:.6}",
                         epoch + 1, epoch_loss, val_loss, directional_acc * 100.0, lr);
            }

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                let metrics = TrainingMetrics {
                    asset: asset.clone(),
                    source: "investing".to_string(),
                    final_loss: epoch_loss,
                    final_val_loss: val_loss,
                    directional_accuracy: directional_acc,
                    mape: calculate_mape(&val_predictions, &val_targets),
                    rmse: calculate_rmse(&val_predictions, &val_targets),
                    epochs_trained: epoch + 1,
                    training_time: start_time.elapsed().as_secs_f64(),
                    timestamp: Utc::now().to_rfc3339(),
                };
                let weights = model.to_weights(&cli, closing_mean, closing_std, opening_mean, opening_std, metrics);
                save_model_to_postgres(&mut pg_client, &weights)?;
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

        let mut train_predictions = Vec::with_capacity(train_sequences.len());
        let mut val_predictions = Vec::with_capacity(val_sequences.len());
        for seq in &train_sequences {
            train_predictions.push(model.forward(seq, false).min(3.0));
        }
        for seq in &val_sequences {
            val_predictions.push(model.forward(seq, false).min(3.0));
        }

        let final_train_loss = calculate_rmse(&train_predictions, &train_targets).powi(2);
        let final_val_loss = calculate_rmse(&val_predictions, &val_targets).powi(2);
        let train_directional_acc = calculate_directional_accuracy(&train_predictions, &train_targets, &train_sequences);
        let val_directional_acc = calculate_directional_accuracy(&val_predictions, &val_targets, &val_sequences);

        let train_pred_denorm: Vec<f32> = train_predictions.iter().map(|&x| x * closing_std + closing_mean).collect();
        let train_targets_denorm: Vec<f32> = train_targets.iter().map(|&x| x * closing_std + closing_mean).collect();
        let val_pred_denorm: Vec<f32> = val_predictions.iter().map(|&x| x * closing_std + closing_mean).collect();
        let val_targets_denorm: Vec<f32> = val_targets.iter().map(|&x| x * closing_std + closing_mean).collect();

        let train_mape = calculate_mape(&train_pred_denorm, &train_targets_denorm);
        let val_mape = calculate_mape(&val_pred_denorm, &val_targets_denorm);
        let train_rmse = calculate_rmse(&train_pred_denorm, &train_targets_denorm);
        let val_rmse = calculate_rmse(&val_pred_denorm, &val_targets_denorm);

        println!("\n✅ Training completed for {}!", asset);
        println!("   📊 Final Results:");
        println!("      🎯 Train Loss: {:.6}", final_train_loss);
        println!("      🎯 Val Loss: {:.6}", final_val_loss);
        println!("      📈 Train Dir Accuracy: {:.1}%", train_directional_acc * 100.0);
        println!("      📈 Val Dir Accuracy: {:.1}%", val_directional_acc * 100.0);
        println!("      📊 Train MAPE: {:.2}%", train_mape * 100.0);
        println!("      📊 Val MAPE: {:.2}%", val_mape * 100.0);
        println!("      📏 Train RMSE: {:.4}", train_rmse);
        println!("      📏 Val RMSE: {:.4}", val_rmse);
        println!("      ⏱️ Training Time: {:.1}s", training_time);
        println!("      🔢 Data Stats: Closing Mean={:.2}, Closing Std={:.2}, Opening Mean={:.2}, Opening Std={:.2}",
                 closing_mean, closing_std, opening_mean, opening_std);

        println!("\n🔮 Sample predictions for {} (last 5 validation samples):", asset);
        let n_val = val_sequences.len();
        for i in (n_val.saturating_sub(5))..n_val {
            let pred = val_predictions[i];
            let actual = val_targets[i];
            let last_closing = val_sequences[i][0];
            let pred_denorm = pred * closing_std + closing_mean;
            let actual_denorm = actual * closing_std + closing_mean;
            let last_denorm = last_closing * closing_std + closing_mean;

            let direction_correct = (pred > last_closing) == (actual > last_closing);
            println!("      Sample {}: Last Closing={:.2} → Pred={:.2}, Actual={:.2} (Error: {:.2}) {}",
                     i + 1, last_denorm, pred_denorm, actual_denorm, (pred_denorm - actual_denorm).abs(),
                     if direction_correct { "✅" } else { "❌" });
        }

        overall_metrics.push((asset, val_directional_acc, val_mape, val_rmse));
    }

    let total_training_time = total_start_time.elapsed().as_secs_f64();
    println!("\n📊 Summary for all assets:");
    println!("   ⏱️ Total Training Time: {:.1}s", total_training_time);
    println!("   🔢 Assets Processed: {}", overall_metrics.len());

    let avg_directional_acc = overall_metrics.iter().map(|(_, acc, _, _)| *acc).sum::<f32>() / overall_metrics.len().max(1) as f32;
    let avg_mape = overall_metrics.iter().map(|(_, _, mape, _)| *mape).sum::<f32>() / overall_metrics.len().max(1) as f32;
    let avg_rmse = overall_metrics.iter().map(|(_, _, _, rmse)| *rmse).sum::<f32>() / overall_metrics.len().max(1) as f32;

    println!("   📈 Average Directional Accuracy: {:.1}%", avg_directional_acc * 100.0);
    println!("   📊 Average MAPE: {:.2}%", avg_mape * 100.0);
    println!("   📏 Average RMSE: {:.4}", avg_rmse);

    println!("\n📈 Per-Asset Performance:");
    for (asset, acc, mape, rmse) in overall_metrics {
        println!("      {}: Dir Acc={:.1}%, MAPE={:.2}%, RMSE={:.4}", asset, acc * 100.0, mape * 100.0, rmse);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();
    info!("Starting Enhanced LSTM training with parameters:");
    info!("  Sequence Length: {}", cli.seq_length);
    info!("  Hidden Size: {}", cli.hidden_size);
    info!("  Number of Layers: {}", cli.num_layers);
    info!("  Learning Rate: {}", cli.learning_rate);
    info!("  Epochs: {}", cli.epochs);
    info!("  Batch Size: {}", cli.batch_size);
    info!("  Dropout Rate: {}", cli.dropout_rate);

    train_model(cli).map_err(|e| {
        error!("Training failed: {}", e);
        Box::new(e) as Box<dyn std::error::Error>
    })
}
 

 



// cd ~/Documentos/projetos/rust_app/lstmfiletrain
// cargo run --release -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 50 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005


 
 

 

 // cargo run --release -- --asset WEGE --num-predictions 5 --verbose



// cd ~/Documentos/projetos/rust_app/lstmfilepredict
// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose

 

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
 