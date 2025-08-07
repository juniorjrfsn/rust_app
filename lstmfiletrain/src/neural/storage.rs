// projeto: lstmfiletrain
// file: src/neural/storage.rs
// Handles storage of model weights and training metrics to PostgreSQL.






use postgres::Client;
use serde_json;
use crate::neural::model::ModelWeights;
use crate::neural::metrics::TrainingMetrics;
use crate::neural::utils::TrainingError;

pub fn ensure_tables_exist(client: &mut Client) -> Result<(), TrainingError> {
    println!("🔍 [Storage] Checking if required tables exist...");

    let create_stock_records = r#"
        CREATE TABLE IF NOT EXISTS stock_records (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(100) NOT NULL,
            date VARCHAR(20) NOT NULL,
            opening REAL NOT NULL,
            closing REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            volume REAL NOT NULL,
            variation REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_asset_date UNIQUE(asset, date)
        )
    "#;

    let create_model_weights = r#"
        CREATE TABLE IF NOT EXISTS model_weights (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(100) NOT NULL,
            weights_json TEXT NOT NULL,
            timestamp VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_asset_timestamp_weights UNIQUE(asset, timestamp)
        )
    "#;

    let create_training_metrics = r#"
        CREATE TABLE IF NOT EXISTS training_metrics (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(100) NOT NULL,
            source VARCHAR(50) NOT NULL,
            train_loss DOUBLE PRECISION NOT NULL,
            val_loss DOUBLE PRECISION NOT NULL,
            rmse DOUBLE PRECISION NOT NULL,
            mae DOUBLE PRECISION NOT NULL,
            mape DOUBLE PRECISION NOT NULL,
            directional_accuracy DOUBLE PRECISION NOT NULL,
            timestamp VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_asset_timestamp_metrics UNIQUE(asset, timestamp)
        )
    "#;

    // First, try to alter existing tables if they exist with the old schema
    let alter_commands = vec![
        "ALTER TABLE stock_records ALTER COLUMN asset TYPE VARCHAR(100)",
        "ALTER TABLE model_weights ALTER COLUMN asset TYPE VARCHAR(100)",
        "ALTER TABLE training_metrics ALTER COLUMN asset TYPE VARCHAR(100)",
    ];

    // Try to alter tables, but ignore errors if tables don't exist or columns are already correct
    for alter_cmd in alter_commands {
        match client.execute(alter_cmd, &[]) {
            Ok(_) => println!("✅ [Storage] Successfully altered table column"),
            Err(e) => println!("ℹ️ [Storage] Alter table info (may be expected): {}", e),
        }
    }

    let create_indexes = vec![
        "CREATE INDEX IF NOT EXISTS idx_stock_records_asset_date ON stock_records(asset, date)",
        "CREATE INDEX IF NOT EXISTS idx_model_weights_asset ON model_weights(asset)",
        "CREATE INDEX IF NOT EXISTS idx_model_weights_timestamp ON model_weights(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_training_metrics_asset ON training_metrics(asset)",
        "CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp ON training_metrics(timestamp)",
    ];

    let create_update_function = r#"
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    "#;

    let create_triggers = vec![
        "DROP TRIGGER IF EXISTS update_model_weights_updated_at ON model_weights",
        r#"CREATE TRIGGER update_model_weights_updated_at 
           BEFORE UPDATE ON model_weights 
           FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()"#,
        "DROP TRIGGER IF EXISTS update_training_metrics_updated_at ON training_metrics",
        r#"CREATE TRIGGER update_training_metrics_updated_at 
           BEFORE UPDATE ON training_metrics 
           FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()"#,
    ];

    client.batch_execute(create_stock_records)?;
    client.batch_execute(create_model_weights)?;
    client.batch_execute(create_training_metrics)?;
    for index in create_indexes {
        client.batch_execute(index)?;
    }
    client.batch_execute(create_update_function)?;
    for trigger in create_triggers {
        client.batch_execute(trigger)?;
    }

    println!("✅ [Storage] All required tables and indexes created successfully.");
    Ok(())
}

pub fn save_model_to_postgres(
    client: &mut Client,
    weights: &ModelWeights,
    metrics: &TrainingMetrics,
) -> Result<(), TrainingError> {
    println!("💾 [Storage] Saving model weights and metrics for asset '{}'", weights.asset);
    ensure_tables_exist(client)?;

    let weights_json = serde_json::to_string(weights)?;
    let exists_weights: i64 = client
        .query_one(
            "SELECT COUNT(*) FROM model_weights WHERE asset = $1 AND timestamp = $2",
            &[&weights.asset, &weights.timestamp],
        )?
        .get(0);

    if exists_weights > 0 {
        client.execute(
            "UPDATE model_weights SET weights_json = $2 WHERE asset = $1 AND timestamp = $3",
            &[&weights.asset, &weights_json, &weights.timestamp],
        )?;
    } else {
        client.execute(
            "INSERT INTO model_weights (asset, weights_json, timestamp) VALUES ($1, $2, $3)",
            &[&weights.asset, &weights_json, &weights.timestamp],
        )?;
    }

    println!("📊 [Storage] Saving metrics to database for asset '{}'", metrics.asset);
    let exists_metrics: i64 = client
        .query_one(
            "SELECT COUNT(*) FROM training_metrics WHERE asset = $1 AND timestamp = $2",
            &[&metrics.asset, &metrics.timestamp],
        )?
        .get(0);

    if exists_metrics > 0 {
        client.execute(
            "UPDATE training_metrics SET source = $2, train_loss = $3, val_loss = $4, rmse = $5, mae = $6, mape = $7, directional_accuracy = $8 WHERE asset = $1 AND timestamp = $9",
            &[
                &metrics.asset,
                &metrics.source,
                &metrics.train_loss,
                &metrics.val_loss,
                &metrics.rmse,
                &metrics.mae,
                &metrics.mape,
                &metrics.directional_accuracy,
                &metrics.timestamp,
            ],
        )?;
    } else {
        client.execute(
            "INSERT INTO training_metrics (asset, source, train_loss, val_loss, rmse, mae, mape, directional_accuracy, timestamp) 
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &metrics.asset,
                &metrics.source,
                &metrics.train_loss,
                &metrics.val_loss,
                &metrics.rmse,
                &metrics.mae,
                &metrics.mape,
                &metrics.directional_accuracy,
                &metrics.timestamp,
            ],
        )?;
    }

    println!("✅ [Storage] Model weights and metrics saved successfully.");
    Ok(())
}

pub fn load_model_from_postgres(
    client: &mut Client,
    asset: &str,
) -> Result<Option<ModelWeights>, TrainingError> {
    ensure_tables_exist(client)?;
    println!("📥 [Storage] Loading model weights for asset '{}'", asset);
    let rows = client.query(
        "SELECT weights_json FROM model_weights WHERE asset = $1 ORDER BY timestamp DESC LIMIT 1",
        &[&asset],
    )?;

    if rows.is_empty() {
        println!("⚠️ [Storage] No model weights found for asset '{}'", asset);
        return Ok(None);
    }

    let weights_json: &str = rows[0].get(0);
    let weights: ModelWeights = serde_json::from_str(weights_json)?;
    println!("✅ [Storage] Model weights loaded successfully for asset '{}'", asset);
    Ok(Some(weights))
}

pub fn load_metrics_from_postgres(
    client: &mut Client,
    asset: &str,
) -> Result<Option<TrainingMetrics>, TrainingError> {
    ensure_tables_exist(client)?;
    println!("📊 [Storage] Loading training metrics for asset '{}'", asset);
    let rows = client.query(
        "SELECT asset, source, train_loss, val_loss, rmse, mae, mape, directional_accuracy, timestamp 
         FROM training_metrics WHERE asset = $1 ORDER BY timestamp DESC LIMIT 1",
        &[&asset],
    )?;

    if rows.is_empty() {
        println!("⚠️ [Storage] No training metrics found for asset '{}'", asset);
        return Ok(None);
    }

    let row = &rows[0];
    let metrics = TrainingMetrics {
        asset: row.get(0),
        source: row.get(1),
        train_loss: row.get(2),
        val_loss: row.get(3),
        rmse: row.get(4),
        mae: row.get(5),
        mape: row.get(6),
        directional_accuracy: row.get(7),
        timestamp: row.get(8),
    };

    println!("✅ [Storage] Training metrics loaded successfully for asset '{}'", asset);
    Ok(Some(metrics))
}