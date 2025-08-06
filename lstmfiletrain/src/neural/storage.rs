
// projeto: lstmfiletrain
// file: src/neural/storage.rs




use postgres::Client;
use serde_json;
use crate::neural::model::ModelWeights;
use crate::neural::metrics::TrainingMetrics;
use crate::neural::utils::TrainingError;

pub fn save_model_to_postgres(client: &mut Client, weights: &ModelWeights, metrics: &TrainingMetrics) -> Result<(), TrainingError> {
    let weights_json = serde_json::to_string(weights)
        .map_err(|e| TrainingError::SerializationError(e.to_string()))?;
    
    client.execute(
        "INSERT INTO lstm_weights (asset, weights_json, created_at) VALUES ($1, $2, $3)
         ON CONFLICT (asset) DO UPDATE SET weights_json = $2, created_at = $3",
        &[&weights.asset, &weights_json, &weights.timestamp],
    )?;

    client.execute(
        "INSERT INTO training_metrics (asset, source, train_loss, val_loss, rmse, mae, mape, directional_accuracy, created_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         ON CONFLICT (asset) DO UPDATE SET source = $2, train_loss = $3, val_loss = $4, rmse = $5, mae = $6, mape = $7, directional_accuracy = $8, created_at = $9",
        &[&metrics.asset, &metrics.source, &metrics.train_loss, &metrics.val_loss, &metrics.rmse,
          &metrics.mae, &metrics.mape, &metrics.directional_accuracy, &metrics.timestamp],
    )?;

    Ok(())
}