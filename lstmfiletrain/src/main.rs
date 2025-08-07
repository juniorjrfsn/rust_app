// projeto: lstmfiletrain
// file: src/main.rs
// Main entry point for training the LSTM model.

 
 
 
 
 // projeto: lstmfiletrain
// file: src/main.rs
// Main entry point for training the LSTM model.

mod neural;

use clap::Parser;
use chrono::Utc;
use ndarray::Array2;
use crate::neural::data::{connect_db, DataLoader};
use crate::neural::metrics::TrainingMetrics;
use crate::neural::model::LstmPredictor;
use crate::neural::storage::save_model_to_postgres;
use crate::neural::utils::{AdamOptimizer, TrainingError};

fn clean_asset_name(asset: &str) -> String {
    // Remove common suffixes and clean up asset names
    let cleaned = asset
        .replace(" Dados Históricos", "")
        .replace(" Historical Data", "")
        .replace(" Preços Históricos", "")
        .trim()
        .to_string();
    
    // Limit to 50 characters to be safe
    if cleaned.len() > 50 {
        cleaned[..50].to_string()
    } else {
        cleaned
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value_t = String::from("ISAE4"))]
    asset: String,
    #[arg(long)]
    all_assets: bool,
    #[arg(long, default_value_t = 40)]
    seq_length: usize,
    #[arg(long, default_value_t = 64)]
    hidden_size: usize,
    #[arg(long, default_value_t = 2)]
    num_layers: usize,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.3)]
    dropout_rate: f64,
    #[arg(long, default_value_t = 0.0005)]
    learning_rate: f64,
    #[arg(long, default_value_t = 0.01)]
    l2_weight: f64,
    #[arg(long, default_value_t = 1.0)]
    clip_norm: f64,
    #[arg(long, default_value_t = String::from("postgresql://postgres:postgres@localhost:5432/lstm_db"))]
    db_url: String,
}

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();

    if cli.all_assets {
        println!(
            "🚀 [Main] Training for all assets at {}",
            Utc::now().format("%Y-%m-%d %H:%M:%S")
        );
        // Load assets list first
        let assets = {
            let mut client = connect_db(&cli.db_url)?;
            let mut loader = DataLoader::new(&mut client)?;
            loader.load_all_assets()?
        };
        
        // Process each asset with a fresh connection
        for asset in &assets {
            // Clean asset name to remove common suffixes and limit length
            let clean_asset = clean_asset_name(asset);
            println!("📌 [Main] Processing asset: {} (cleaned: {})", asset, clean_asset);
            train_for_asset(&cli, &clean_asset)?;
        }
    } else {
        println!(
            "🚀 [Main] Starting training for asset: {} at {}",
            cli.asset,
            Utc::now().format("%Y-%m-%d %H:%M:%S")
        );
        train_for_asset(&cli, &cli.asset)?;
    }

    println!(
        "✅ [Main] Training completed successfully at {}",
        Utc::now().format("%Y-%m-%d %H:%M:%S")
    );
    Ok(())
}

fn train_for_asset(
    cli: &Cli,
    asset: &str,
) -> Result<(), TrainingError> {
    // Create a fresh connection and loader for this asset
    let mut client = connect_db(&cli.db_url)?;
    let mut loader = DataLoader::new(&mut client)?;
    
    println!("📥 [Main] Loading asset data for '{}'", asset);
    let records = loader.load_asset_data(asset)?;
    println!(
        "✅ [Main] Asset data loaded successfully for '{}', {} records",
        asset, records.len()
    );

    println!(
        "🔧 [Main] Creating sequences with length {} for {} records",
        cli.seq_length, records.len()
    );
    let (train_seqs, train_targets, feature_stats) =
        loader.create_sequences(&records, cli.seq_length)?;
    println!(
        "✅ [Main] Sequences created successfully, {} sequences",
        train_seqs.len()
    );

    println!("🔄 [Main] Converting data to f64 and splitting into train/validation sets...");
    let mut train_seqs: Vec<Array2<f64>> = train_seqs
        .iter()
        .map(|seq| seq.mapv(|x| x as f64))
        .collect();
    let mut train_targets: Vec<f64> = train_targets.iter().map(|&x| x as f64).collect();

    let split_index = (train_seqs.len() as f64 * 0.8) as usize;
    let mut val_seqs: Vec<Array2<f64>> = train_seqs.split_off(split_index);
    let mut val_targets: Vec<f64> = train_targets.split_off(split_index);

    println!("🔄 [Main] Normalizing data...");
    for seq in &mut train_seqs {
        for i in 0..seq.ncols() {
            let col_mean = feature_stats.feature_means[i] as f64;
            let col_std = feature_stats.feature_stds[i] as f64;
            seq.column_mut(i).mapv_inplace(|x| (x - col_mean) / col_std);
        }
    }

    for seq in &mut val_seqs {
        for i in 0..seq.ncols() {
            let col_mean = feature_stats.feature_means[i] as f64;
            let col_std = feature_stats.feature_stds[i] as f64;
            seq.column_mut(i).mapv_inplace(|x| (x - col_mean) / col_std);
        }
    }

    let closing_mean = feature_stats.closing_mean as f64;
    let closing_std = feature_stats.closing_std as f64;

    for target in &mut train_targets {
        *target = (*target - closing_mean) / closing_std;
    }

    for target in &mut val_targets {
        *target = (*target - closing_mean) / closing_std;
    }

    println!(
        "✅ [Main] Data normalized and split: {} train, {} validation sequences",
        train_seqs.len(),
        val_seqs.len()
    );

    println!(
        "🛠️ [Main] Initializing LSTM model with hidden_size: {}, num_layers: {}",
        cli.hidden_size, cli.num_layers
    );
    let mut model = LstmPredictor::new(
        feature_stats.feature_names.len(),
        cli.hidden_size,
        cli.num_layers,
        cli.dropout_rate,
    )?;
    println!("✅ [Main] LSTM model initialized successfully");

    println!(
        "🛠️ [Main] Initializing Adam optimizer with learning_rate: {}",
        cli.learning_rate
    );
    let mut optimizer = AdamOptimizer::new(cli.learning_rate, 0.9, 0.999, 1e-8);
    println!("✅ [Main] Adam optimizer initialized successfully");

    for epoch in 1..=cli.epochs {
        println!("🌱 [Main] Epoch {}/{}", epoch, cli.epochs);

        println!("🎓 [Main] Performing training step...");
        let train_loss = model.train_step(
            &train_seqs,
            &train_targets,
            &mut optimizer,
            cli.l2_weight,
            cli.clip_norm,
        )?;
        println!("📉 [Main] Train Loss: {:.6}", train_loss);

        println!("🔮 [Main] Performing validation...");
        let val_predictions: Result<Vec<f64>, TrainingError> = val_seqs
            .iter()
            .map(|seq| model.predict(seq))
            .collect();
        let val_predictions = val_predictions?;

        let val_loss = val_predictions
            .iter()
            .zip(val_targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / val_predictions.len() as f64;
        println!("📊 [Main] Validation Loss: {:.6}", val_loss);

        let rmse = val_loss.sqrt();
        let mae = val_predictions
            .iter()
            .zip(val_targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / val_predictions.len() as f64;

        if epoch % 10 == 0 {
            println!("💾 [Main] Saving model and metrics for epoch {}...", epoch);
            let mut weights = model.get_weights();
            weights.closing_mean = closing_mean;
            weights.closing_std = closing_std;
            weights.asset = asset.to_string();
            weights.seq_length = cli.seq_length;

            let metrics = TrainingMetrics {
                asset: asset.to_string(),
                source: "database".to_string(),
                train_loss,
                val_loss,
                rmse,
                mae,
                mape: 0.0, // Placeholder, requires implementation
                directional_accuracy: 0.0, // Placeholder, requires implementation
                timestamp: Utc::now().to_rfc3339(),
            };

            // Create a fresh connection for saving since we consumed the previous one
            let mut save_client = connect_db(&cli.db_url)?;
            if let Err(e) = save_model_to_postgres(&mut save_client, &weights, &metrics) {
                println!("⚠️ [Main] Warning: Could not save model: {}", e);
            } else {
                println!(
                    "✅ [Main] Model and metrics saved successfully for epoch {}",
                    epoch
                );
            }
        }
    }

    Ok(())
}


// cd ~/Documentos/projetos/rust_app/lstmfiletrain
// cargo run --release -- --asset ISAE4 --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --learning-rate 0.001
 // cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 
// cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 --db-url postgres://postgres:postgres@localhost:5432/lstm_db

 // cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 --db-url postgres://postgres:postgres@localhost:5432/lstm_db
 // cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 --db-url postgresql://postgres:postgres@localhost:5432/lstm_db

 // cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0
 // rm -rf target Cargo.lock
//  cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 --db-url postgresql://postgres:postgres@localhost:5432/lstm_db


// rm -rf target Cargo.lock
// cargo run --release -- --all-assets --seq-length 20 --hidden-size 32 --num-layers 2 --epochs 10 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.001 --l2-weight 0.01 --clip-norm 1.0 --db-url postgresql://postgres:postgres@localhost:5432/lstm_db


// cargo run --release -- --asset ISAE4 --seq-length 40 --epochs 10

 

// cd lstmfiletrain
 

 

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
 