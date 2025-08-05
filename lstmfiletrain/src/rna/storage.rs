// projeto: lstmfiletrain
// file: src/rna/storage.rs


// projeto: lstmfiletrain
// file: src/rna/storage.rs

use crate::{Cli, TrainingMetrics, LSTMError};
use crate::model::MultiLayerLSTM;
use postgres::{Client, Transaction}; // Keeping Transaction for now, consider removing if unused
use serde_json;
use log::info;

pub fn save_model_to_postgres(
    pg_client: &mut Client,
    model: &MultiLayerLSTM,
    cli: &Cli,
    asset: &str,
    closing_mean: f32,
    closing_std: f32,
    opening_mean: f32,
    opening_std: f32,
    metrics: &TrainingMetrics,
) -> Result<(), LSTMError> {
    let mut transaction = pg_client.transaction()?; // Made mutable
    transaction.batch_execute(
        "CREATE TABLE IF NOT EXISTS lstm_weights_v3 (
            asset TEXT NOT NULL,
            source TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset, source)
        )",
    )?;

    transaction.batch_execute(
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
        )",
    )?;

    let mut weights = model.to_weights(cli, closing_mean, closing_std, opening_mean, opening_std);
    weights.asset = asset.to_string();
    let weights_json = serde_json::to_string(&weights)
        .map_err(|e| LSTMError::TrainingError(format!("Failed to serialize weights: {}", e)))?;

    transaction.execute(
        "INSERT INTO lstm_weights_v3 (asset, source, weights_json) 
         VALUES ($1, $2, $3)
         ON CONFLICT ON CONSTRAINT lstm_weights_v3_pkey 
         DO UPDATE SET weights_json = $3, created_at = CURRENT_TIMESTAMP",
        &[&asset, &weights.source, &weights_json],
    )?;

    transaction.execute(
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
        &[
            &metrics.asset,
            &metrics.source,
            &metrics.final_loss,
            &metrics.final_val_loss,
            &metrics.directional_accuracy,
            &metrics.mape,
            &metrics.rmse,
            &(metrics.epochs_trained as i32),
            &metrics.training_time,
        ],
    )?;

    info!("Model and metrics saved to PostgreSQL for asset {}", asset);
    transaction.commit()?; // Ensure transaction is committed
    Ok(())
}