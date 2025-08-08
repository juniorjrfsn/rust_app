// projeto: lstmrnntrain
// file: src/main.rs
// Main entry point for multi-model deep learning training system

 
 
 

mod neural;

use clap::Parser;
use chrono::Utc;
use log::{info, error, debug};
use ndarray::Array2;
use std::time::Instant;

use crate::neural::data::{connect_db, ensure_tables_exist, DataLoader};
use crate::neural::metrics::{TrainingMetrics, MetricsTracker};
use crate::neural::model::{ModelType, NeuralNetwork};
use crate::neural::storage::save_model_to_postgres;
use crate::neural::utils::{AdamOptimizer, TrainingError, LearningRateScheduler};

/// Cleans and normalizes asset names by removing common suffixes
fn clean_asset_name(asset: &str) -> String {
    let cleaned = asset
        .replace(" Dados Históricos", "")
        .replace(" Historical Data", "")
        .replace(" Preços Históricos", "")
        .replace(" - Investing.com", "")
        .replace("(", "")
        .replace(")", "")
        .trim()
        .to_string();
    
    // Limit to 80 characters for database compatibility
    if cleaned.len() > 80 {
        cleaned[..80].to_string()
    } else {
        cleaned
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "lstmrnntrain",
    author = "AI Trading Systems",
    version = "2.0.0",
    about = "Neural network training system for financial asset price prediction",
    long_about = "Advanced system supporting multiple models (LSTM, RNN, MLP, CNN) for financial time series analysis and prediction with performance optimizations and regularization."
)]
struct Cli {
    /// Specific asset to train (e.g., PETR4, VALE3)
    #[arg(long, default_value_t = String::from("PETR4"))]
    asset: String,

    /// Train models for all available assets
    #[arg(long)]
    all_assets: bool,

    /// Neural network model type
    #[arg(long, value_enum, default_value = "lstm")]
    model_type: ModelTypeArg,

    /// Input sequence length
    #[arg(long, default_value_t = 40, help = "Number of historical days for prediction")]
    seq_length: usize,

    /// Hidden layer size
    #[arg(long, default_value_t = 128, help = "Number of neurons in hidden layers")]
    hidden_size: usize,

    /// Number of model layers
    #[arg(long, default_value_t = 3, help = "Number of deep layers")]
    num_layers: usize,

    /// Number of training epochs
    #[arg(long, default_value_t = 100, help = "Complete iterations over the dataset")]
    epochs: usize,

    /// Batch size for training
    #[arg(long, default_value_t = 64, help = "Number of samples per batch")]
    batch_size: usize,

    /// Dropout rate for regularization
    #[arg(long, default_value_t = 0.3, help = "Dropout rate (0.0-1.0)")]
    dropout_rate: f64,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.001, help = "Learning rate for Adam optimizer")]
    learning_rate: f64,

    /// L2 regularization weight
    #[arg(long, default_value_t = 0.01, help = "L2 regularization weight")]
    l2_weight: f64,

    /// Maximum norm for gradient clipping
    #[arg(long, default_value_t = 1.0, help = "Maximum norm for gradient clipping")]
    clip_norm: f64,

    /// Early stopping patience
    #[arg(long, default_value_t = 15, help = "Epochs without improvement to stop")]
    patience: usize,

    /// Train/validation split ratio
    #[arg(long, default_value_t = 0.8, help = "Proportion of data for training")]
    train_split: f64,

    /// PostgreSQL connection URL
    #[arg(long, default_value_t = String::from("postgresql://postgres:postgres@localhost:5432/rnn_db"))]
    db_url: String,

    /// Verbose logging mode
    #[arg(long)]
    verbose: bool,

    /// Save checkpoints periodically
    #[arg(long, default_value_t = 10, help = "Save frequency (epochs)")]
    save_freq: usize,

    /// Use learning rate scheduler
    #[arg(long)]
    use_scheduler: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum ModelTypeArg {
    Lstm,
    Rnn,
    Mlp,
    Cnn,
}

impl From<ModelTypeArg> for ModelType {
    fn from(arg: ModelTypeArg) -> Self {
        match arg {
            ModelTypeArg::Lstm => ModelType::LSTM,
            ModelTypeArg::Rnn => ModelType::RNN,
            ModelTypeArg::Mlp => ModelType::MLP,
            ModelTypeArg::Cnn => ModelType::CNN,
        }
    }
}

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    
    // Setup logging
    setup_logging(cli.verbose);
    
    let start_time = Instant::now();
    let model_type = ModelType::from(cli.model_type.clone());
    
    info!("🚀 Deep Learning System started");
    info!("📊 Model: {:?} | Sequence: {} | Hidden: {} | Layers: {}", 
          model_type, cli.seq_length, cli.hidden_size, cli.num_layers);
    info!("🕐 Started at: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"));

    // Setup database
    setup_database(&cli.db_url)?;

    let result = if cli.all_assets {
        train_all_assets(&cli, model_type)
    } else {
        let clean_asset = clean_asset_name(&cli.asset);
        info!("🎯 Training single asset: {} (cleaned: {})", cli.asset, clean_asset);
        train_single_asset(&cli, &clean_asset, model_type)
    };

    let elapsed = start_time.elapsed();
    match result {
        Ok(_) => {
            info!("✅ Training completed successfully in {:.2}s", elapsed.as_secs_f64());
            info!("🏁 Finished at: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"));
        }
        Err(e) => {
            error!("❌ Error during training: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn setup_logging(verbose: bool) {
    let level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::from_default_env()
        .filter_level(level)
        .format_timestamp_secs()
        .init();
}

fn setup_database(db_url: &str) -> Result<(), TrainingError> {
    info!("🔧 Setting up database");
    let mut client = connect_db(db_url)?;
    ensure_tables_exist(&mut client)?;
    info!("✅ Database configured");
    Ok(())
}

fn train_all_assets(cli: &Cli, model_type: ModelType) -> Result<(), TrainingError> {
    info!("🔄 Starting training for all assets");
    
    // Load list of assets
    let assets = {
        let mut client = connect_db(&cli.db_url)?;
        let mut loader = DataLoader::new(&mut client)?;
        loader.load_all_assets()?
    };
    
    info!("📋 Found {} unique assets", assets.len());
    
    let mut successful_trains = 0;
    let mut failed_trains = 0;
    
    for (idx, asset) in assets.iter().enumerate() {
        info!("🔄 Training asset {}/{}: {}", idx + 1, assets.len(), asset);
        match train_single_asset(cli, asset, model_type) {
            Ok(_) => successful_trains += 1,
            Err(e) => {
                failed_trains += 1;
                error!("❌ Failed training for asset {}: {}", asset, e);
            }
        }
    }
    
    info!("✅ Training complete: {} successful, {} failed", successful_trains, failed_trains);
    Ok(())
}

fn train_single_asset(cli: &Cli, asset: &str, model_type: ModelType) -> Result<(), TrainingError> {
    let training_start = Instant::now();
    info!("📊 Starting training for asset: {}", asset);
    
    // Initialize database connection and data loader
    let mut client = connect_db(&cli.db_url)?;
    let mut loader = DataLoader::new(&mut client)?;
    
    // Load and preprocess data
    let records = loader.load_asset_data(asset)?;
    let (sequences, targets, feature_stats) = loader.create_sequences(&records, cli.seq_length)?;
    
    // Split and normalize data
    let ((train_seqs, train_targets), (val_seqs, val_targets)) = 
        preprocess_data(sequences, targets, &feature_stats, cli.train_split)?;
    
    // Initialize model and optimizer
    let mut model = NeuralNetwork::new(
        model_type,
        feature_stats.feature_names.len(),
        cli.hidden_size,
        cli.num_layers,
        cli.dropout_rate,
        cli.seq_length, // Added the missing seq_length parameter
    )?;
    
    let mut optimizer = AdamOptimizer::new(cli.learning_rate, 0.9, 0.999, 1e-8);
    let scheduler = if cli.use_scheduler {
        Some(LearningRateScheduler::StepDecay {
            initial_rate: cli.learning_rate,
            decay_rate: 0.5,
            step_size: 10,
        })
    } else {
        None
    };
    
    let mut metrics_tracker = MetricsTracker::new();
    let mut best_val_loss = f64::INFINITY;
    
    // Training loop
    for epoch in 1..=cli.epochs {
        let epoch_start = Instant::now();
        
        // Update learning rate if scheduler is used
        if let Some(scheduler) = &scheduler {
            let new_lr = scheduler.get_rate(epoch);
            optimizer.set_learning_rate(new_lr);
            debug!("🔄 Epoch {}: Learning rate set to {}", epoch, new_lr);
        }
        
        // Train
        let train_loss = model.train(
            &train_seqs,
            &train_targets,
            &mut optimizer,
            cli.batch_size,
            cli.l2_weight,
            cli.clip_norm,
        )?;

        // Validate
        let (val_loss, val_metrics) = model.validate(&val_seqs, &val_targets)?;
        
        let epoch_time = epoch_start.elapsed().as_secs_f64();
        
        // Log progress
        if epoch % 5 == 0 || epoch <= 10 {
            info!("📈 Epoch {}/{}: Train={:.6} | Val={:.6} | RMSE={:.6} | R²={:.4} | {:.1}s", 
                  epoch, cli.epochs, train_loss, val_loss, val_metrics.rmse, 
                  val_metrics.r_squared, epoch_time);
        }

        // Create metrics for tracker
        let metrics = TrainingMetrics {
            asset: asset.to_string(),
            model_type: format!("{:?}", model_type),
            source: "database".to_string(),
            epoch,
            train_loss,
            val_loss,
            rmse: val_metrics.rmse,
            mae: val_metrics.mae,
            mape: val_metrics.mape,
            directional_accuracy: val_metrics.directional_accuracy,
            r_squared: val_metrics.r_squared,
            timestamp: Utc::now().to_rfc3339(),
        };

        // Early stopping check
        let should_stop = metrics_tracker.add_metrics(metrics.clone(), cli.patience);
        
        // Save model if improved or at specified frequency
        if val_loss < best_val_loss || epoch % cli.save_freq == 0 {
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                info!("🎯 New best validation loss: {:.6}", best_val_loss);
            }
            
            debug!("💾 Saving model at epoch {}", epoch);
            save_model_and_metrics(
                &cli.db_url, 
                &model, 
                asset, 
                model_type,
                &feature_stats,
                cli.seq_length,
                train_loss,
                val_loss,
                &val_metrics,
                epoch
            )?;
        }

        // Early stopping
        if should_stop {
            info!("⏹️ Early stopping triggered at epoch {}", epoch);
            break;
        }
    }

    let training_time = training_start.elapsed().as_secs_f64();
    
    // Final summary
    metrics_tracker.print_summary();
    info!("⏱️ Training of '{}' completed in {:.2}s", asset, training_time);
    
    Ok(())
}

fn preprocess_data(
    train_seqs: Vec<Array2<f32>>,
    train_targets: Vec<f32>,
    feature_stats: &crate::neural::data::FeatureStats,
    split_ratio: f64,
) -> Result<((Vec<Array2<f64>>, Vec<f64>), (Vec<Array2<f64>>, Vec<f64>)), TrainingError> {
    
    debug!("🔄 Converting data to f64 and normalizing");
    
    // Convert to f64 and normalize
    let mut train_seqs: Vec<Array2<f64>> = train_seqs
        .iter()
        .map(|seq| {
            let mut seq_f64 = seq.mapv(|x| x as f64);
            normalize_sequence(&mut seq_f64, feature_stats);
            seq_f64
        })
        .collect();
        
    let mut train_targets: Vec<f64> = train_targets.iter().map(|&x| x as f64).collect();

    // Normalize targets
    let closing_mean = feature_stats.closing_mean as f64;
    let closing_std = feature_stats.closing_std as f64;
    
    for target in &mut train_targets {
        *target = (*target - closing_mean) / closing_std;
    }

    // Split data
    let split_index = (train_seqs.len() as f64 * split_ratio) as usize;
    let val_seqs: Vec<Array2<f64>> = train_seqs.split_off(split_index);
    let val_targets: Vec<f64> = train_targets.split_off(split_index);

    debug!("✅ Data normalized and split");
    Ok(((train_seqs, train_targets), (val_seqs, val_targets)))
}

fn normalize_sequence(seq: &mut Array2<f64>, feature_stats: &crate::neural::data::FeatureStats) {
    for i in 0..seq.ncols() {
        if i < feature_stats.feature_means.len() && i < feature_stats.feature_stds.len() {
            let col_mean = feature_stats.feature_means[i] as f64;
            let col_std = feature_stats.feature_stds[i] as f64;
            if col_std > 1e-8 {
                seq.column_mut(i).mapv_inplace(|x| (x - col_mean) / col_std);
            }
        }
    }
}

fn save_model_and_metrics(
    db_url: &str,
    model: &NeuralNetwork,
    asset: &str,
    model_type: ModelType,
    feature_stats: &crate::neural::data::FeatureStats,
    seq_length: usize,
    train_loss: f64,
    val_loss: f64,
    val_metrics: &crate::neural::model::ValidationMetrics,
    epoch: usize,
) -> Result<(), TrainingError> {
    
    debug!("💾 Saving model and metrics");
    let mut save_client = connect_db(db_url)?;
    let mut weights = model.get_weights();
    
    // Configure model metadata
    weights.asset = asset.to_string();
    weights.model_type = model_type;
    weights.closing_mean = feature_stats.closing_mean as f64;
    weights.closing_std = feature_stats.closing_std as f64;
    weights.seq_length = seq_length;
    weights.epoch = epoch;
    weights.timestamp = Utc::now().to_rfc3339();

    let metrics = TrainingMetrics {
        asset: asset.to_string(),
        model_type: format!("{:?}", model_type),
        source: "database".to_string(),
        epoch,
        train_loss,
        val_loss,
        rmse: val_metrics.rmse,
        mae: val_metrics.mae,
        mape: val_metrics.mape,
        directional_accuracy: val_metrics.directional_accuracy,
        r_squared: val_metrics.r_squared,
        timestamp: Utc::now().to_rfc3339(),
    };

    save_model_to_postgres(&mut save_client, &weights, &metrics)?;
    debug!("✅ Model and metrics saved successfully");
    Ok(())
}

// Example usage commands:
// cargo run --release -- --model-type lstm --asset ISAE4 --seq-length 30 --epochs 50 --verbose
// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64  
// cargo run --release -- --model-type mlp --asset PETR4 --hidden-size 128 --epochs 100
// cargo run --release -- --model-type cnn --all-assets --seq-length 40 --num-layers 3


// cargo run --release -- --model-type rnn --asset PETR4 --seq-length 20 --epochs 30 --batch-size 64 --verbose 
// cargo run --release -- --model-type rnn --asset PETR4 --seq-length 20 --epochs 50 --batch-size 32 --hidden-size 64 --num-layers 2 --dropout-rate 0.2 --learning-rate 0.001 --patience 15 --use-scheduler --verbose
 

// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64
// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64 --verbose

// cargo run --release -- --model-type rnn --asset PETR4 --seq-length 20 --epochs 30 --batch-size 64 --verbose


//# 1. Treinar RNN com um asset específico e verbose ativado
//cargo run --release -- --model-type rnn --asset PETR4 --seq-length 20 --epochs 30 --batch-size 32 --verbose

//# 2. Treinar RNN com todos os assets e mais detalhes
//cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64 --verbose --patience 10

//# 3. Treinar RNN com scheduler de learning rate
//cargo run --release -- --model-type rnn --asset VALE3 --seq-length 20 --epochs 50 --batch-size 32 --use-scheduler --verbose
//
//# 4. Treinar RNN com parâmetros otimizados
// cargo run --release -- --model-type rnn --all-assets  --seq-length 20   --epochs 50   --batch-size 32   --hidden-size 64   --num-layers 2    --dropout-rate 0.2   --learning-rate 0.001   --l2-weight 0.01   --patience 15  --use-scheduler  --verbose






//# 5. Comparar RNN com outros modelos
//echo "=== Treinando RNN ==="
//cargo run --release -- --model-type rnn --asset PETR4 --epochs 30 --verbose

//echo "=== Treinando LSTM ==="
//cargo run --release -- --model-type lstm --asset PETR4 --epochs 30 --verbose

//echo "=== Treinando MLP ==="
//cargo run --release -- --model-type mlp --asset PETR4 --epochs 30 --verbose





// cargo run --release -- --model-type lstm --asset PETR4 --seq-length 40 --epochs 100 --verbose
// cargo run --release -- --model-type rnn --all-assets --seq-length 30 --epochs 50 --batch-size 128 --use-scheduler
// cargo run --release -- --model-type mlp --asset VALE3 --hidden-size 256 --num-layers 4 --patience 20
// cargo run --release -- --model-type cnn --all-assets --dropout-rate 0.4 --l2-weight 0.001

// Example usage commands:
// cargo run --release -- --model-type lstm --asset ISAE4 --seq-length 30 --epochs 50 --verbose
// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64
// cargo run --release -- --model-type mlp --asset PETR4 --hidden-size 128 --epochs 100
// cargo run --release -- --model-type cnn --all-assets --seq-length 40 --num-layers 3


// # Single asset training with RNN
// cargo run --release -- --model-type rnn --asset PETR4 --seq-length 20 --epochs 30 --batch-size 64

// # All assets training
// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64

// # With verbose logging
// cargo run --release -- --model-type lstm --asset VALE3 --seq-length 30 --epochs 50 --verbose

// # Different model types
// cargo run --release -- --model-type mlp --asset PETR4 --hidden-size 128 --epochs 100
// cargo run --release -- --model-type cnn --all-assets --seq-length 40 --num-layers 3