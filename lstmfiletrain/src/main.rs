// projeto: lstmfiletrain
// file: src/main.rs
 
 


use clap::Parser;
use log::{info, warn};
use postgres::{Client, NoTls};
use std::error::Error;
// Import TrainingError from its correct location within the crate
use crate::neural::utils::TrainingError; 
use crate::neural::data::DataLoader;
use crate::neural::model::LstmPredictor;
use crate::neural::metrics::TrainingMetrics;
use crate::neural::storage::save_model_to_postgres;
use crate::neural::utils; // For AdamOptimizer, metrics functions

mod neural;
 

#[derive(Parser)]
#[command(name = "lstm_train")]
#[command(about = "Train LSTM model for stock price prediction")]
struct Cli {
    // Make asset optional and add a flag for all assets
    #[arg(long, help = "Asset symbol (e.g., WEGE3). If omitted, --all-assets is assumed unless a list is provided via stdin/file (future enhancement).")]
    asset: Option<String>,

    #[arg(long, default_value_t = false, help = "Train models for all assets found in the database")]
    all_assets: bool,

    #[arg(long, default_value_t = 20, help = "Sequence length")]
    seq_length: usize,

    #[arg(long, default_value_t = 50, help = "Hidden size")]
    hidden_size: usize,

    #[arg(long, default_value_t = 2, help = "Number of LSTM layers")]
    num_layers: usize,

    #[arg(long, default_value_t = 0.0001, help = "Learning rate")]
    learning_rate: f32,

    #[arg(long, default_value_t = 100, help = "Number of epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 32, help = "Batch size (currently unused, processes full dataset per epoch)")]
    batch_size: usize, // Note: Batch size is parsed but not used in current train_step

    #[arg(long, default_value_t = 0.1, help = "Dropout rate")]
    dropout_rate: f32,

    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db", help = "Database URL")]
    db_url: String,

    #[arg(long, default_value_t = 0.01, help = "L2 Regularization weight")]
    l2_weight: f32,

    #[arg(long, default_value_t = 1.0, help = "Gradient clipping norm")]
    clip_norm: f32,
}

// Function to encapsulate the training logic for a single asset
// Returns Result<(), TrainingError> to allow using ? with TrainingError
fn train_single_asset(
    cli: &Cli, // Pass the CLI struct to access parameters
    client: &mut Client,
    asset: &str,
    source: &str, // Source can be derived or fixed, e.g., "database"
) -> Result<(), TrainingError> { // Changed return type to TrainingError
    info!("Iniciando treinamento para o ativo: {}", asset);

    // Load data for the specific asset
    let mut data_loader = DataLoader::new(client)?; // Pass &mut Client reference
    let records = data_loader.load_asset_data(asset)?; // load_asset_data now uses LIKE

    if records.is_empty() {
        warn!("Nenhum dado encontrado para o ativo: {}. Pulando.", asset);
        return Ok(());
    }

    let (sequences, targets, feature_stats) = data_loader.create_sequences(&records, cli.seq_length)?;

    // Split into train and validation
    let train_size = (sequences.len() as f32 * 0.8) as usize;
    if train_size == 0 || train_size >= sequences.len() {
         return Err(TrainingError::DataProcessing(
            format!("Dados insuficientes após divisão para o ativo: {}", asset)
        ));
    }
    let (train_seqs, val_seqs) = sequences.split_at(train_size);
    let (train_targets, val_targets) = targets.split_at(train_size);

    // Check if validation set is empty
    if val_seqs.is_empty() || val_targets.is_empty() {
         return Err(TrainingError::DataProcessing(
            format!("Conjunto de validação vazio para o ativo: {}", asset)
        ));
    }

    // Initialize model
    let mut model = LstmPredictor::new(
        feature_stats.feature_names.len(),
        cli.hidden_size,
        cli.num_layers,
        cli.dropout_rate,
    )?;
    let mut optimizer = utils::AdamOptimizer::new(model.num_parameters(), cli.learning_rate, 0.9, 0.999, 1e-8);

    // Training loop
    let mut best_val_loss = f32::INFINITY;
    let mut best_weights = None;
    let mut best_metrics = None;

    for epoch in 1..=cli.epochs {
        // Perform training step on the entire training set (consider batching)
        let train_loss = model.train_step(&train_seqs, &train_targets, &mut optimizer, cli.l2_weight, cli.clip_norm)?;

        // --- CORRECTED SECTION ---
        // Predict on the actual validation set
        let val_predictions: Result<Vec<f32>, TrainingError> = val_seqs // Use val_seqs, not train_seqs
            .iter()
            .map(|seq| model.predict(seq)) // model.predict can return TrainingError
            .collect();
        let val_predictions = val_predictions?; // Handle potential prediction errors

        // Calculate validation loss using validation predictions and targets
        let val_loss = utils::mse_loss(&val_predictions, &val_targets);
        // --- END CORRECTED SECTION ---

        info!("Ativo {}: Epoch {}: Train Loss = {:.6}, Val Loss = {:.6}", asset, epoch, train_loss, val_loss);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;

            // Get model weights
            let mut weights = model.get_weights();
            weights.asset = asset.to_string(); // Set asset in weights
            weights.seq_length = cli.seq_length;
            weights.closing_mean = feature_stats.closing_mean;
            weights.closing_std = feature_stats.closing_std;

            // --- Use consistent timestamp ---
            let current_timestamp = chrono::Utc::now().to_rfc3339();
            // --- End timestamp ---

            // Calculate metrics using validation predictions and targets
            let metrics = TrainingMetrics {
                asset: asset.to_string(),
                source: source.to_string(),
                train_loss,
                val_loss,
                rmse: val_loss.sqrt(),
                mae: utils::mae_loss(&val_predictions, &val_targets),
                mape: utils::mape_loss(&val_predictions, &val_targets),
                directional_accuracy: utils::directional_accuracy(&val_predictions, &val_targets),
                timestamp: current_timestamp, // Use consistent timestamp
            };

            best_weights = Some(weights);
            best_metrics = Some(metrics);
            // Note: Saving inside the loop is expensive. We save only the best model at the end.
        }
    }

    // Save the best model and metrics found during training
    if let (Some(weights), Some(metrics)) = (best_weights, best_metrics) {
        info!("Salvando melhor modelo para o ativo {} com Val Loss = {:.6}", asset, best_val_loss);
        // Pass &mut Client reference
        save_model_to_postgres(client, &weights, &metrics)?;
    } else {
        warn!("Nenhum modelo melhor encontrado para salvar para o ativo {}", asset);
    }

    info!("✅ Treinamento concluído para o ativo {}", asset);
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let mut client = Client::connect(&cli.db_url, NoTls)?;

    // --- CORRECTED VARIABLE DECLARATION AND LOGIC ---
    // Determine assets_to_train based on CLI arguments
    let assets_to_train: Vec<String> = if cli.all_assets {
        info!("Modo: Treinar todos os ativos");
        let all_assets_full = DataLoader::new(&mut client)?.load_all_assets()?;
        if all_assets_full.is_empty() {
            warn!("Nenhum ativo encontrado no banco de dados.");
            return Ok(());
        }
        // Extract symbols
        let extracted_symbols: Vec<String> = all_assets_full
            .into_iter()
            .filter_map(|full_name| {
                full_name.split_whitespace().next().map(|s| s.to_string())
            })
            .collect();
        if extracted_symbols.is_empty() {
             warn!("Nenhum símbolo de ativo pôde ser extraído dos nomes encontrados.");
             return Ok(());
        }
        extracted_symbols // Return the list directly

    } else if let Some(asset) = cli.asset.as_ref() {
        info!("Modo: Treinar ativo específico: {}", asset);
        vec![asset.clone()] // Return a vector with one element

    } else {
        // Default behavior if neither --asset nor --all-assets is provided
        info!("Nenhum ativo especificado explicitamente. Treinando todos os ativos.");
        let all_assets_full = DataLoader::new(&mut client)?.load_all_assets()?;
         if all_assets_full.is_empty() {
            warn!("Nenhum ativo encontrado no banco de dados.");
            return Ok(());
        }
        let extracted_symbols: Vec<String> = all_assets_full
            .into_iter()
            .filter_map(|full_name| {
                full_name.split_whitespace().next().map(|s| s.to_string())
            })
            .collect();
         if extracted_symbols.is_empty() {
             warn!("Nenhum símbolo de ativo pôde ser extraído dos nomes encontrados.");
             return Ok(());
        }
        extracted_symbols // Return the list directly
    };
    // --- END CORRECTED VARIABLE DECLARATION AND LOGIC ---

    let source = "database";

    info!("Iniciando treinamento para {} ativo(s): {:?}", assets_to_train.len(), assets_to_train);

    let mut success_count = 0;
    let mut failure_count = 0;

    for asset in &assets_to_train {
        // Pass &mut client reference and handle TrainingError
        match train_single_asset(&cli, &mut client, asset, source) {
            Ok(()) => {
                success_count += 1;
                info!("Treinamento bem-sucedido para: {}", asset);
            }
            Err(e) => {
                failure_count += 1;
                // Log the TrainingError
                eprintln!("Erro de treinamento para {}: {}", asset, e);
                // Consider logging to a file or database
            }
        }
    }

    info!(
        "🏁 Processo de treinamento finalizado. Sucessos: {}, Falhas: {}",
        success_count, failure_count
    );

    if failure_count > 0 {
        // Optionally return an error if any training failed, or just log and exit(0)
        // For now, we'll log and exit successfully, as some assets might fail legitimately (e.g., insufficient data)
         warn!("{} ativo(s) falharam durante o treinamento.", failure_count);
         // If you want the main program to exit with an error code if any training failed:
         // return Err(format!("{} ativo(s) falharam durante o treinamento.", failure_count).into());
    }

    Ok(())
}





// cd ~/Documentos/projetos/rust_app/lstmfiletrain

// cargo run --release -- --all-assets --seq-length 20 --hidden-size 50 --num-layers 2 --learning-rate 0.0001 --epochs 100 --batch-size 32 --dropout-rate 0.1 --l2-weight 0.01 --clip-norm 1.0 --db-url postgres://postgres:postgres@localhost:5432/lstm_db
 

 
// cargo run --release -- --asset WEGE3 --source investing --seq-length 20 --hidden-size 50 --num-layers 2 --learning-rate 0.0001 --epochs 100 --batch-size 32 --db-url postgres://postgres:postgres@localhost:5432/lstm_db

 


// cargo run --release -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 50 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005


 // cargo run --release -- --all-assets --seq-length 20 --hidden-size 50 --num-layers 2 --learning-rate 0.0001 --epochs 100 --batch-size 32 --dropout-rate 0.1 --l2-weight 0.01 --clip-norm 1.0 --db-url postgres://postgres:postgres@localhost:5432/lstm_db
 
 




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
 