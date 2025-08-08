// projeto: lstmrnntrain
// file: src/neural/storage.rs
// Model and metrics storage logic for PostgreSQL




// projeto: lstmrnntrain
// file: src/neural/storage.rs
// Model and metrics storage logic for PostgreSQL

use postgres::Client;
use crate::neural::model::ModelWeights;
use crate::neural::metrics::TrainingMetrics;
use crate::neural::utils::TrainingError;

pub fn save_model_to_postgres(
    client: &mut Client,
    weights: &ModelWeights,
    metrics: &TrainingMetrics,
) -> Result<(), TrainingError> {
    println!("💾 [Storage] Saving model weights and metrics to PostgreSQL");
    
    let mut transaction = client.transaction()?;
    
    // Serialize weights using serde_json as a fallback for complex structures
    let weights_data = serde_json::to_string(weights)
        .map_err(|e| TrainingError::Serialization(format!("Failed to serialize weights: {}", e)))?;
    
    // Insert model weights
    let weights_query = "
        INSERT INTO model_weights 
        (asset, model_type, weights_data, closing_mean, closing_std, 
         seq_length, hidden_size, num_layers, feature_dim, epoch)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ";
    
    transaction.execute(
        weights_query,
        &[
            &weights.asset,
            &format!("{:?}", weights.model_type),
            &weights_data,
            &weights.closing_mean,
            &weights.closing_std,
            &(weights.seq_length as i32),
            &(weights.hidden_size as i32),
            &(weights.num_layers as i32),
            &(weights.feature_dim as i32),
            &(weights.epoch as i32),
        ],
    )?;
    
    // Insert training metrics
    let metrics_query = "
        INSERT INTO training_metrics 
        (asset, model_type, source, epoch, train_loss, val_loss, rmse, mae, mape, 
         directional_accuracy, r_squared, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ";
    
    transaction.execute(
        metrics_query,
        &[
            &metrics.asset,
            &metrics.model_type,
            &metrics.source,
            &(metrics.epoch as i32),
            &metrics.train_loss,
            &metrics.val_loss,
            &metrics.rmse,
            &metrics.mae,
            &metrics.mape,
            &metrics.directional_accuracy,
            &metrics.r_squared,
            &metrics.timestamp,
        ],
    )?;
    
    transaction.commit()?;
    
    println!("✅ [Storage] Model and metrics saved successfully");
    println!("   - Asset: {}", weights.asset);
    println!("   - Model Type: {:?}", weights.model_type);
    println!("   - Epoch: {}", weights.epoch);
    println!("   - Validation Loss: {:.6}", metrics.val_loss);
    
    Ok(())
}

pub fn load_model_from_postgres(
    client: &mut Client,
    asset: &str,
    model_type: &str,
) -> Result<(ModelWeights, TrainingMetrics), TrainingError> {
    println!("📂 [Storage] Loading model weights and metrics from PostgreSQL");
    println!("   - Asset: {}", asset);
    println!("   - Model Type: {}", model_type);
    
    // Load the most recent model weights
    let weights_query = "
        SELECT weights_data, closing_mean, closing_std, seq_length, 
               hidden_size, num_layers, feature_dim, epoch, created_at
        FROM model_weights 
        WHERE asset = $1 AND model_type = $2 
        ORDER BY created_at DESC 
        LIMIT 1
    ";
    
    let weights_row = client.query_one(weights_query, &[&asset, &model_type])
        .map_err(|e| TrainingError::DataProcessing(
            format!("No model found for asset '{}' and type '{}': {}", asset, model_type, e)
        ))?;
    
    // Deserialize weights using serde_json
    let weights_data: String = weights_row.get("weights_data");
    let mut weights: ModelWeights = serde_json::from_str(&weights_data)
        .map_err(|e| TrainingError::Serialization(format!("Failed to deserialize weights: {}", e)))?;
    
    // Update metadata from database
    weights.closing_mean = weights_row.get("closing_mean");
    weights.closing_std = weights_row.get("closing_std");
    weights.seq_length = weights_row.get::<_, i32>("seq_length") as usize;
    weights.hidden_size = weights_row.get::<_, i32>("hidden_size") as usize;
    weights.num_layers = weights_row.get::<_, i32>("num_layers") as usize;
    weights.feature_dim = weights_row.get::<_, i32>("feature_dim") as usize;
    weights.epoch = weights_row.get::<_, i32>("epoch") as usize;
    weights.asset = asset.to_string();
    
    // Load corresponding metrics
    let metrics_query = "
        SELECT asset, model_type, source, epoch, train_loss, val_loss, rmse, mae, mape,
               directional_accuracy, r_squared, timestamp
        FROM training_metrics 
        WHERE asset = $1 AND model_type = $2 AND epoch = $3
        ORDER BY created_at DESC 
        LIMIT 1
    ";
    
    let metrics_row = client.query_one(metrics_query, &[&asset, &model_type, &(weights.epoch as i32)])
        .map_err(|e| TrainingError::DataProcessing(
            format!("No metrics found for asset '{}', type '{}', epoch {}: {}", 
                    asset, model_type, weights.epoch, e)
        ))?;
    
    let metrics = TrainingMetrics {
        asset: metrics_row.get("asset"),
        model_type: metrics_row.get("model_type"),
        source: metrics_row.get("source"),
        epoch: metrics_row.get::<_, i32>("epoch") as usize,
        train_loss: metrics_row.get("train_loss"),
        val_loss: metrics_row.get("val_loss"),
        rmse: metrics_row.get("rmse"),
        mae: metrics_row.get("mae"),
        mape: metrics_row.get("mape"),
        directional_accuracy: metrics_row.get("directional_accuracy"),
        r_squared: metrics_row.get("r_squared"),
        timestamp: metrics_row.get("timestamp"),
    };
    
    println!("✅ [Storage] Model loaded successfully");
    println!("   - Epoch: {}", weights.epoch);
    println!("   - RMSE: {:.6}", metrics.rmse);
    println!("   - R²: {:.6}", metrics.r_squared);
    
    Ok((weights, metrics))
}

pub fn list_available_models(client: &mut Client) -> Result<Vec<(String, String, i32, f64)>, TrainingError> {
    println!("📋 [Storage] Listing all available models");
    
    let query = "
        SELECT DISTINCT w.asset, w.model_type, w.epoch, m.val_loss
        FROM model_weights w
        LEFT JOIN training_metrics m ON (
            w.asset = m.asset AND 
            w.model_type = m.model_type AND 
            w.epoch = m.epoch
        )
        ORDER BY w.asset, w.model_type, w.epoch DESC
    ";
    
    let rows = client.query(query, &[])?;
    let models: Vec<(String, String, i32, f64)> = rows.iter().map(|row| {
        (
            row.get("asset"),
            row.get("model_type"),
            row.get("epoch"),
            row.get::<_, Option<f64>>("val_loss").unwrap_or(0.0),
        )
    }).collect();
    
    println!("✅ [Storage] Found {} model entries", models.len());
    Ok(models)
}

pub fn get_best_model_for_asset(
    client: &mut Client,
    asset: &str,
) -> Result<(String, f64), TrainingError> {
    println!("🎯 [Storage] Finding best model for asset: {}", asset);
    
    let query = "
        SELECT w.model_type, m.val_loss, m.rmse, m.r_squared
        FROM model_weights w
        JOIN training_metrics m ON (
            w.asset = m.asset AND 
            w.model_type = m.model_type AND 
            w.epoch = m.epoch
        )
        WHERE w.asset = $1
        ORDER BY m.val_loss ASC
        LIMIT 1
    ";
    
    let row = client.query_one(query, &[&asset])
        .map_err(|e| TrainingError::DataProcessing(
            format!("No models found for asset '{}': {}", asset, e)
        ))?;
    
    let model_type: String = row.get("model_type");
    let val_loss: f64 = row.get("val_loss");
    let rmse: f64 = row.get("rmse");
    let r_squared: f64 = row.get("r_squared");
    
    println!("✅ [Storage] Best model found:");
    println!("   - Type: {}", model_type);
    println!("   - Validation Loss: {:.6}", val_loss);
    println!("   - RMSE: {:.6}", rmse);
    println!("   - R²: {:.6}", r_squared);
    
    Ok((model_type, val_loss))
}

pub fn delete_old_models(
    client: &mut Client,
    asset: &str,
    keep_best_n: usize,
) -> Result<usize, TrainingError> {
    println!("🧹 [Storage] Cleaning up old models for asset: {}", asset);
    
    let mut transaction = client.transaction()?;
    
    // Get models to keep (best N by validation loss)
    let keep_query = "
        SELECT w.id as weight_id, m.id as metric_id
        FROM model_weights w
        JOIN training_metrics m ON (
            w.asset = m.asset AND 
            w.model_type = m.model_type AND 
            w.epoch = m.epoch
        )
        WHERE w.asset = $1
        ORDER BY m.val_loss ASC
        LIMIT $2
    ";
    
     
    
   
    
 

    let keep_rows = transaction.query(keep_query, &[&asset, &(keep_best_n as i32)])?;
    let _keep_weight_ids: Vec<i32> = keep_rows.iter().map(|row| row.get("weight_id")).collect();
    let _keep_metric_ids: Vec<i32> = keep_rows.iter().map(|row| row.get("metric_id")).collect(); // Prefixed with underscore

    let mut deleted_count = 0;
    
    if !keep_weight_ids.is_empty() {
        // Delete old weights - using a simpler approach
        let delete_weights_query = "
            DELETE FROM model_weights 
            WHERE asset = $1 AND id NOT IN (
                SELECT w.id FROM model_weights w
                JOIN training_metrics m ON (
                    w.asset = m.asset AND 
                    w.model_type = m.model_type AND 
                    w.epoch = m.epoch
                )
                WHERE w.asset = $1
                ORDER BY m.val_loss ASC
                LIMIT $2
            )
        ";
        
        deleted_count += transaction.execute(delete_weights_query, &[&asset, &(keep_best_n as i32)])?;
        
        // Delete old metrics
        let delete_metrics_query = "
            DELETE FROM training_metrics 
            WHERE asset = $1 AND id NOT IN (
                SELECT m.id FROM training_metrics m
                JOIN model_weights w ON (
                    w.asset = m.asset AND 
                    w.model_type = m.model_type AND 
                    w.epoch = m.epoch
                )
                WHERE m.asset = $1
                ORDER BY m.val_loss ASC
                LIMIT $2
            )
        ";
        
        deleted_count += transaction.execute(delete_metrics_query, &[&asset, &(keep_best_n as i32)])?;
    }
    
    transaction.commit()?;
    
    println!("✅ [Storage] Cleaned up {} old records for asset: {}", deleted_count, asset);
    Ok(deleted_count as usize)
}

pub fn export_model_performance(
    client: &mut Client,
    output_file: Option<&str>,
) -> Result<(), TrainingError> {
    println!("📊 [Storage] Exporting model performance summary");
    
    let query = "
        SELECT 
            w.asset,
            w.model_type,
            COUNT(*) as total_epochs,
            MIN(m.val_loss) as best_val_loss,
            MAX(m.r_squared) as best_r_squared,
            AVG(m.rmse) as avg_rmse,
            AVG(m.directional_accuracy) as avg_direction_acc,
            MAX(w.created_at) as latest_training
        FROM model_weights w
        JOIN training_metrics m ON (
            w.asset = m.asset AND 
            w.model_type = m.model_type AND 
            w.epoch = m.epoch
        )
        GROUP BY w.asset, w.model_type
        ORDER BY w.asset, best_val_loss ASC
    ";
    
    let rows = client.query(query, &[])?;
    
    let mut summary = String::new();
    summary.push_str("Asset,Model Type,Total Epochs,Best Val Loss,Best R²,Avg RMSE,Avg Direction Acc,Latest Training\n");
    
    for row in rows {
        summary.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.4},{}\n",
            row.get::<_, String>("asset"),
            row.get::<_, String>("model_type"),
            row.get::<_, i64>("total_epochs"),
            row.get::<_, f64>("best_val_loss"),
            row.get::<_, f64>("best_r_squared"),
            row.get::<_, f64>("avg_rmse"),
            row.get::<_, f64>("avg_direction_acc"),
            row.get::<_, chrono::DateTime<chrono::Utc>>("latest_training").format("%Y-%m-%d %H:%M:%S"),
        ));
    }
    
    match output_file {
        Some(file_path) => {
            std::fs::write(file_path, summary)?;
            println!("✅ [Storage] Performance summary exported to: {}", file_path);
        },
        None => {
            println!("📊 [Storage] Model Performance Summary:");
            println!("{}", summary);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::model::{ModelType, ModelWeights, MlpLayerWeights};
    use crate::neural::metrics::TrainingMetrics;
    use ndarray::{Array1, Array2};
    use chrono::Utc;

    #[test]
    fn test_model_weights_serialization() {
        let weights = ModelWeights {
            asset: "TEST".to_string(),
            model_type: ModelType::LSTM,
            lstm_layers: vec![],
            rnn_layers: vec![],
            mlp_layers: vec![],
            cnn_layers: vec![],
            final_layer: MlpLayerWeights {
                w: Array2::zeros((1, 10)),
                b: Array1::zeros(1),
            },
            closing_mean: 100.0,
            closing_std: 10.0,
            seq_length: 20,
            hidden_size: 64,
            num_layers: 2,
            feature_dim: 8,
            epoch: 50,
            timestamp: Utc::now().to_rfc3339(),
        };
        
        // Test serialization with serde_json
        let serialized = serde_json::to_string(&weights);
        assert!(serialized.is_ok());
        
        // Test deserialization
        let deserialized: Result<ModelWeights, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
        
        let restored_weights = deserialized.unwrap();
        assert_eq!(restored_weights.asset, "TEST");
        assert_eq!(restored_weights.model_type, ModelType::LSTM);
        assert_eq!(restored_weights.seq_length, 20);
    }

    #[test]
    fn test_training_metrics_structure() {
        let metrics = TrainingMetrics {
            asset: "TEST".to_string(),
            model_type: "LSTM".to_string(),
            source: "database".to_string(),
            epoch: 50,
            train_loss: 0.001,
            val_loss: 0.0015,
            rmse: 0.05,
            mae: 0.03,
            mape: 2.5,
            directional_accuracy: 0.67,
            r_squared: 0.85,
            timestamp: Utc::now().to_rfc3339(),
        };
        
        assert_eq!(metrics.asset, "TEST");
        assert_eq!(metrics.model_type, "LSTM");
        assert_eq!(metrics.epoch, 50);
        assert!(metrics.r_squared > 0.0);
        assert!(metrics.directional_accuracy > 0.0);
    }
}