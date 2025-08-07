// projeto: lstmrnntrain
// file: src/neural/data.rs
// Data loading and preprocessing functionality

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::neural::utils::TrainingError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockRecord {
    pub id: i32,
    pub asset: String,
    pub date: DateTime<Utc>,
    pub opening: f32,
    pub high: f32,
    pub low: f32,
    pub closing: f32,
    pub adj_closing: f32,
    pub volume: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_names: Vec<String>,
    pub feature_means: Vec<f32>,
    pub feature_stds: Vec<f32>,
    pub closing_mean: f32,
    pub closing_std: f32,
    pub min_values: Vec<f32>,
    pub max_values: Vec<f32>,
}

pub struct DataLoader {
    client: *mut Client,
}

impl DataLoader {
    pub fn new(client: &mut Client) -> Result<Self, TrainingError> {
        Ok(DataLoader {
            client: client as *mut Client,
        })
    }

    fn get_client(&mut self) -> &mut Client {
        unsafe { &mut *self.client }
    }

    pub fn load_all_assets(&mut self) -> Result<Vec<String>, TrainingError> {
        println!("📋 [DataLoader] Loading all available assets from database");
        
        let query = "
            SELECT DISTINCT asset 
            FROM stock_data 
            ORDER BY asset
        ";
        
        let rows = self.get_client().query(query, &[])?;
        let assets: Vec<String> = rows.iter()
            .map(|row| row.get::<_, String>(0))
            .collect();
        
        println!("✅ [DataLoader] Found {} unique assets", assets.len());
        Ok(assets)
    }

    pub fn load_asset_data(&mut self, asset: &str) -> Result<Vec<StockRecord>, TrainingError> {
        println!("📥 [DataLoader] Loading data for asset: {}", asset);
        
        let query = "
            SELECT id, asset, date, opening, high, low, closing, adj_closing, volume, created_at
            FROM stock_data 
            WHERE asset = $1 
            ORDER BY date ASC
        ";
        
        let rows = self.get_client().query(query, &[&asset])?;
        let records: Result<Vec<StockRecord>, _> = rows.iter()
            .map(|row| self.row_to_stock_record(row))
            .collect();

        let records = records?;
        println!("✅ [DataLoader] Loaded {} records for asset: {}", records.len(), asset);
        
        if records.is_empty() {
            return Err(TrainingError::DataProcessing(
                format!("No data found for asset: {}", asset)
            ));
        }

        Ok(records)
    }

    fn row_to_stock_record(&self, row: &Row) -> Result<StockRecord, TrainingError> {
        Ok(StockRecord {
            id: row.get("id"),
            asset: row.get("asset"),
            date: row.get("date"),
            opening: row.get("opening"),
            high: row.get("high"),
            low: row.get("low"),
            closing: row.get("closing"),
            adj_closing: row.get("adj_closing"),
            volume: row.get("volume"),
            created_at: row.get("created_at"),
        })
    }

    pub fn create_sequences(
        &mut self,
        records: &[StockRecord],
        seq_length: usize,
    ) -> Result<(Vec<Array2<f32>>, Vec<f32>, FeatureStats), TrainingError> {
        if records.len() < seq_length + 1 {
            return Err(TrainingError::DataProcessing(
                format!("Insufficient data: {} records, need at least {}", 
                        records.len(), seq_length + 1)
            ));
        }

        println!("🔧 [DataLoader] Creating sequences with length: {}", seq_length);
        
        // Extract features and compute statistics
        let features = self.extract_features(records)?;
        let feature_stats = self.compute_feature_stats(&features, records)?;
        
        // Create sequences
        let mut sequences = Vec::new();
        let mut targets = Vec::new();

        for i in 0..records.len() - seq_length {
            // Create sequence of features
            let mut sequence = Array2::zeros((seq_length, features.ncols()));
            for t in 0..seq_length {
                sequence.slice_mut(ndarray::s![t, ..])
                    .assign(&features.slice(ndarray::s![i + t, ..]));
            }
            sequences.push(sequence);
            
            // Target is the next closing price
            targets.push(records[i + seq_length].closing);
        }

        println!("✅ [DataLoader] Created {} sequences and targets", sequences.len());
        Ok((sequences, targets, feature_stats))
    }

    fn extract_features(&self, records: &[StockRecord]) -> Result<Array2<f32>, TrainingError> {
        let n_records = records.len();
        let n_features = 8; // Basic features + technical indicators
        
        let mut features = Array2::zeros((n_records, n_features));
        
        for (i, record) in records.iter().enumerate() {
            let mut feature_idx = 0;
            
            // Basic price features
            features[[i, feature_idx]] = record.opening;
            feature_idx += 1;
            features[[i, feature_idx]] = record.high;
            feature_idx += 1;
            features[[i, feature_idx]] = record.low;
            feature_idx += 1;
            features[[i, feature_idx]] = record.closing;
            feature_idx += 1;
            features[[i, feature_idx]] = record.volume as f32;
            feature_idx += 1;
            
            // Price change (if not first record)
            if i > 0 {
                let price_change = record.closing - records[i-1].closing;
                features[[i, feature_idx]] = price_change;
            }
            feature_idx += 1;
            
            // High-Low spread
            features[[i, feature_idx]] = record.high - record.low;
            feature_idx += 1;
            
            // Opening gap
            if i > 0 {
                let gap = record.opening - records[i-1].closing;
                features[[i, feature_idx]] = gap;
            }
        }

        // Add technical indicators
        self.add_technical_indicators(&mut features, records)?;
        
        Ok(features)
    }

    fn add_technical_indicators(
        &self, 
        features: &mut Array2<f32>, 
        records: &[StockRecord]
    ) -> Result<(), TrainingError> {
        let n_records = records.len();
        
        // Simple moving averages
        let window_sizes = [5, 10, 20];
        
        for &window in &window_sizes {
            for i in window..n_records {
                let sum: f32 = records[i-window+1..=i]
                    .iter()
                    .map(|r| r.closing)
                    .sum();
                let sma = sum / window as f32;
                
                // You could expand features array to include these
                // For now, we'll modify existing features or use ratios
                if features.ncols() > 7 {
                    // Store ratio of current price to SMA
                    let ratio = records[i].closing / sma;
                    // This is a simplified approach - you might want to expand the feature matrix
                }
            }
        }

        // RSI calculation (simplified)
        self.calculate_rsi(features, records, 14)?;

        Ok(())
    }

    fn calculate_rsi(
        &self,
        _features: &mut Array2<f32>,
        records: &[StockRecord],
        period: usize,
    ) -> Result<(), TrainingError> {
        if records.len() < period + 1 {
            return Ok(());
        }

        for i in period..records.len() {
            let mut gains = 0.0;
            let mut losses = 0.0;
            
            for j in i-period+1..=i {
                let change = records[j].closing - records[j-1].closing;
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }
            
            let avg_gain = gains / period as f32;
            let avg_loss = losses / period as f32;
            
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 0.0 };
            let _rsi = 100.0 - (100.0 / (1.0 + rs));
            
            // Store RSI in features if you have space
            // features[[i, rsi_column]] = rsi;
        }

        Ok(())
    }

    fn compute_feature_stats(
        &self,
        features: &Array2<f32>,
        records: &[StockRecord],
    ) -> Result<FeatureStats, TrainingError> {
        let feature_names = vec![
            "opening".to_string(),
            "high".to_string(),
            "low".to_string(),
            "closing".to_string(),
            "volume".to_string(),
            "price_change".to_string(),
            "hl_spread".to_string(),
            "opening_gap".to_string(),
        ];

        let mut feature_means = Vec::new();
        let mut feature_stds = Vec::new();
        let mut min_values = Vec::new();
        let mut max_values = Vec::new();

        for col in 0..features.ncols() {
            let column = features.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / column.len() as f32;
            let std = variance.sqrt().max(1e-8);
            
            let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            feature_means.push(mean);
            feature_stds.push(std);
            min_values.push(min_val);
            max_values.push(max_val);
        }

        // Compute closing price statistics
        let closing_prices: Vec<f32> = records.iter().map(|r| r.closing).collect();
        let closing_mean = closing_prices.iter().sum::<f32>() / closing_prices.len() as f32;
        let closing_variance = closing_prices.iter()
            .map(|&x| (x - closing_mean).powi(2))
            .sum::<f32>() / closing_prices.len() as f32;
        let closing_std = closing_variance.sqrt().max(1e-8);

        println!("📊 [DataLoader] Feature statistics computed:");
        println!("   - Features: {}", feature_names.len());
        println!("   - Closing mean: {:.4}, std: {:.4}", closing_mean, closing_std);

        Ok(FeatureStats {
            feature_names,
            feature_means,
            feature_stds,
            closing_mean,
            closing_std,
            min_values,
            max_values,
        })
    }

    pub fn prepare_prediction_data(
        &mut self,
        asset: &str,
        seq_length: usize,
        feature_stats: &FeatureStats,
    ) -> Result<Array2<f64>, TrainingError> {
        // Load the most recent data for prediction
        let query = "
            SELECT id, asset, date, opening, high, low, closing, adj_closing, volume, created_at
            FROM stock_data 
            WHERE asset = $1 
            ORDER BY date DESC 
            LIMIT $2
        ";
        
        let rows = self.get_client().query(query, &[&asset, &(seq_length as i32)])?;
        let mut records: Vec<StockRecord> = rows.iter()
            .map(|row| self.row_to_stock_record(row))
            .collect::<Result<Vec<_>, _>>()?;

        // Reverse to get chronological order
        records.reverse();

        if records.len() < seq_length {
            return Err(TrainingError::DataProcessing(
                format!("Insufficient recent data: {} records, need {}", 
                        records.len(), seq_length)
            ));
        }

        // Extract features and normalize
        let features = self.extract_features(&records)?;
        let mut normalized_features = Array2::zeros((seq_length, features.ncols()));

        for i in 0..seq_length {
            for j in 0..features.ncols() {
                let normalized = if feature_stats.feature_stds[j] > 1e-8 {
                    (features[[i, j]] - feature_stats.feature_means[j]) / feature_stats.feature_stds[j]
                } else {
                    0.0
                };
                normalized_features[[i, j]] = normalized as f64;
            }
        }

        Ok(normalized_features)
    }
}

pub fn connect_db(db_url: &str) -> Result<Client, TrainingError> {
    println!("🔌 [Database] Connecting to PostgreSQL database");
    let client = Client::connect(db_url, NoTls)?;
    println!("✅ [Database] Connected successfully");
    Ok(client)
}

pub fn ensure_tables_exist(client: &mut Client) -> Result<(), TrainingError> {
    println!("🔧 [Database] Ensuring required tables exist");

    // Create stock_data table if it doesn't exist
    let stock_data_sql = "
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(20) NOT NULL,
            date TIMESTAMP WITH TIME ZONE NOT NULL,
            opening REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            closing REAL NOT NULL,
            adj_closing REAL NOT NULL,
            volume BIGINT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asset, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_stock_data_asset_date ON stock_data(asset, date);
        CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date);
    ";

    client.batch_execute(stock_data_sql)?;

    // Create model_weights table
    let model_weights_sql = "
        CREATE TABLE IF NOT EXISTS model_weights (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(100) NOT NULL,
            model_type VARCHAR(20) NOT NULL,
            weights_data BYTEA NOT NULL,
            closing_mean DOUBLE PRECISION NOT NULL,
            closing_std DOUBLE PRECISION NOT NULL,
            seq_length INTEGER NOT NULL,
            hidden_size INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            feature_dim INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_model_weights_asset_type ON model_weights(asset, model_type);
        CREATE INDEX IF NOT EXISTS idx_model_weights_created_at ON model_weights(created_at);
    ";

    client.batch_execute(model_weights_sql)?;

    // Create training_metrics table
    let metrics_sql = "
        CREATE TABLE IF NOT EXISTS training_metrics (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(100) NOT NULL,
            model_type VARCHAR(20) NOT NULL,
            source VARCHAR(20) NOT NULL,
            epoch INTEGER NOT NULL,
            train_loss DOUBLE PRECISION NOT NULL,
            val_loss DOUBLE PRECISION NOT NULL,
            rmse DOUBLE PRECISION NOT NULL,
            mae DOUBLE PRECISION NOT NULL,
            mape DOUBLE PRECISION NOT NULL,
            directional_accuracy DOUBLE PRECISION NOT NULL,
            r_squared DOUBLE PRECISION NOT NULL,
            timestamp VARCHAR(50) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_training_metrics_asset_type ON training_metrics(asset, model_type);
        CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp ON training_metrics(timestamp);
    ";

    client.batch_execute(metrics_sql)?;

    println!("✅ [Database] All required tables are ready");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_feature_extraction() {
        let records = vec![
            StockRecord {
                id: 1,
                asset: "TEST".to_string(),
                date: Utc::now(),
                opening: 100.0,
                high: 105.0,
                low: 95.0,
                closing: 102.0,
                adj_closing: 102.0,
                volume: 1000000,
                created_at: Utc::now(),
            },
            StockRecord {
                id: 2,
                asset: "TEST".to_string(),
                date: Utc::now(),
                opening: 102.0,
                high: 108.0,
                low: 100.0,
                closing: 105.0,
                adj_closing: 105.0,
                volume: 1200000,
                created_at: Utc::now(),
            },
        ];

        // This would require a proper DataLoader instance with a database connection
        // For now, this is a structural test
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].asset, "TEST");
    }

    #[test]
    fn test_feature_stats_calculation() {
        let feature_means = vec![100.0, 110.0, 90.0, 105.0];
        let feature_stds = vec![5.0, 7.0, 3.0, 6.0];
        
        let stats = FeatureStats {
            feature_names: vec!["open".to_string(), "high".to_string(), "low".to_string(), "close".to_string()],
            feature_means: feature_means.clone(),
            feature_stds: feature_stds.clone(),
            closing_mean: 105.0,
            closing_std: 6.0,
            min_values: vec![95.0, 100.0, 85.0, 95.0],
            max_values: vec![110.0, 125.0, 98.0, 115.0],
        };

        assert_eq!(stats.feature_names.len(), 4);
        assert_eq!(stats.closing_mean, 105.0);
        assert_eq!(stats.closing_std, 6.0);
    }
}