// projeto: lstmfiletrain
// file: src/rna/storage.rs
 

 
use crate::LSTMError;
use crate::rna::model::ModelWeights;
use postgres::Client;
use serde_json;
use log::info;

pub fn save_model_to_postgres(
    pg_client: &mut Client,
    weights: &ModelWeights,
) -> Result<(), LSTMError> {
    let mut transaction = pg_client.transaction()
        .map_err(|e| LSTMError::PgError(e))?;
    
    transaction.batch_execute(
        "CREATE TABLE IF NOT EXISTS lstm_weights_v3 (
            asset TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset)
        )",
    )?;

    transaction.batch_execute(
        "CREATE TABLE IF NOT EXISTS training_metrics_v3 (
            asset TEXT NOT NULL,
            final_loss REAL,
            final_val_loss REAL,
            directional_accuracy REAL,
            mape REAL,
            rmse REAL,
            epochs_trained INTEGER,
            training_time DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (asset)
        )",
    )?;

    let weights_json = serde_json::to_string(&weights)
        .map_err(|e| LSTMError::SerializationError(format!("Failed to serialize weights: {}", e)))?;

    transaction.execute(
        "INSERT INTO lstm_weights_v3 (asset, weights_json) 
         VALUES ($1, $2)
         ON CONFLICT ON CONSTRAINT lstm_weights_v3_pkey 
         DO UPDATE SET weights_json = $2, created_at = CURRENT_TIMESTAMP",
        &[&weights.asset, &weights_json],
    )?;

    transaction.execute(
        "INSERT INTO training_metrics_v3 (asset, final_loss, final_val_loss, directional_accuracy, mape, rmse, epochs_trained, training_time) 
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT ON CONSTRAINT training_metrics_v3_pkey 
         DO UPDATE SET 
             final_loss = $2, 
             final_val_loss = $3, 
             directional_accuracy = $4, 
             mape = $5, 
             rmse = $6, 
             epochs_trained = $7, 
             training_time = $8, 
             created_at = CURRENT_TIMESTAMP",
        &[
            &weights.metrics.asset,
            &weights.metrics.final_loss,
            &weights.metrics.final_val_loss,
            &weights.metrics.directional_accuracy,
            &weights.metrics.mape,
            &weights.metrics.rmse,
            &(weights.metrics.epochs_trained as i32),
            &weights.metrics.training_time,
        ],
    )?;

    info!("Model and metrics saved to PostgreSQL for asset {}", weights.metrics.asset);
    transaction.commit()
        .map_err(|e| LSTMError::PgError(e))?;
    Ok(())
}
