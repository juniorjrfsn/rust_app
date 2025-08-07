// projeto: lstmfiletrain
// file: src/neural/data.rs
// Handles data loading and processing from the database.



 
use chrono::NaiveDate;
use log::info;
use ndarray::{Array1, Array2, Axis};
use postgres::{Client, NoTls};
use serde::{Serialize, Deserialize};
use crate::neural::storage::ensure_tables_exist;
use crate::neural::utils::TrainingError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockRecord {
    pub date: NaiveDate,
    pub opening: f32,
    pub closing: f32,
    pub high: f32,
    pub low: f32,
    pub volume: f32,
    pub variation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_means: Vec<f32>,
    pub feature_stds: Vec<f32>,
    pub feature_names: Vec<String>,
    pub closing_mean: f32,
    pub closing_std: f32,
}

pub struct DataLoader<'a> {
    client: &'a mut Client,
}

impl<'a> DataLoader<'a> {
    pub fn new(client: &'a mut Client) -> Result<Self, TrainingError> {
        println!("🔧 [DataLoader] Initializing new DataLoader...");
        println!("🔍 [DataLoader] Ensuring database tables exist...");
        ensure_tables_exist(client)?;
        let loader = DataLoader { client };
        println!("✅ [DataLoader] DataLoader initialized successfully.");
        Ok(loader)
    }

    pub fn load_all_assets(&mut self) -> Result<Vec<String>, TrainingError> {
        println!("📥 [DataLoader] Loading all distinct assets...");
        let rows = self.client.query("SELECT DISTINCT asset FROM stock_records", &[])?;
        let assets: Vec<String> = rows.into_iter().map(|row| row.get(0)).collect();
        println!("✅ [DataLoader] Found {} distinct assets", assets.len());
        Ok(assets)
    }

    pub fn load_asset_data(&mut self, asset: &str) -> Result<Vec<StockRecord>, TrainingError> {
        println!("📥 [DataLoader] Loading data for asset '{}'", asset);
        let query = "SELECT date, opening, closing, high, low, volume, variation FROM stock_records WHERE asset LIKE $1 ORDER BY date ASC";
        let rows = self.client.query(query, &[&format!("%{}%", asset)])?;

        println!("✅ [DataLoader] Found {} rows for '{}'", rows.len(), asset);
        if rows.is_empty() {
            println!("⚠️ [DataLoader] No data found for '{}'.", asset);
        }

        let mut records = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            match NaiveDate::parse_from_str(row.get::<_, &str>(0), "%d.%m.%Y") {
                Ok(parsed_date) => {
                    records.push(StockRecord {
                        date: parsed_date,
                        opening: row.get(1),
                        closing: row.get(2),
                        high: row.get(3),
                        low: row.get(4),
                        volume: row.get(5),
                        variation: row.get(6),
                    });
                    if i < 3 {
                        let r = records.last().unwrap();
                        println!(
                            "📋 [DataLoader] Sample data {}: {} - Open: {}, Close: {}",
                            i + 1,
                            r.date,
                            r.opening,
                            r.closing
                        );
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error parsing date '{}' for {}: {}", row.get::<_, &str>(0), asset, e);
                    println!("❌ [DataLoader] Error: {}", err_msg);
                    info!("{}", err_msg);
                    return Err(TrainingError::DataProcessing(err_msg));
                }
            }
        }

        println!("✅ [DataLoader] Processed {} records for '{}'", records.len(), asset);
        info!("Success: Loaded {} records for {}", records.len(), asset);
        Ok(records)
    }

    pub fn create_sequences(
        &mut self,
        records: &[StockRecord],
        seq_length: usize,
    ) -> Result<(Vec<Array2<f32>>, Vec<f32>, FeatureStats), TrainingError> {
        println!("🔧 [DataLoader] Creating sequences with length {} for {} records", seq_length, records.len());
        if records.len() < seq_length + 1 {
            let err_msg = format!("Not enough records ({}) for sequence length {}", records.len(), seq_length);
            println!("❌ [DataLoader] {}", err_msg);
            return Err(TrainingError::DataProcessing(err_msg));
        }

        let feature_names = vec![
            "opening".to_string(),
            "closing".to_string(),
            "high".to_string(),
            "low".to_string(),
            "volume".to_string(),
            "variation".to_string(),
        ];
        let num_features = feature_names.len();
        println!("📊 [DataLoader] Number of features: {}", num_features);

        let mut feature_matrix = Array2::zeros((records.len(), num_features));
        for (i, record) in records.iter().enumerate() {
            feature_matrix[[i, 0]] = record.opening;
            feature_matrix[[i, 1]] = record.closing;
            feature_matrix[[i, 2]] = record.high;
            feature_matrix[[i, 3]] = record.low;
            feature_matrix[[i, 4]] = record.volume;
            feature_matrix[[i, 5]] = record.variation;
        }

        let feature_means = match feature_matrix.mean_axis(Axis(0)) {
            Some(means) => {
                let means_vec = means.to_vec();
                println!("📊 [DataLoader] Feature means: {:?}", means_vec);
                means_vec
            }
            None => {
                let msg = format!("Warning: No means calculated, using zeros for {} features", num_features);
                println!("⚠️ [DataLoader] {}", msg);
                info!("{}", msg);
                Array1::zeros(num_features).to_vec()
            }
        };

        let feature_stds = feature_matrix.std_axis(Axis(0), 0.0).to_vec();
        println!("📊 [DataLoader] Feature standard deviations: {:?}", feature_stds);

        let closing_mean = feature_means[1];
        let closing_std = feature_stds[1];
        println!(
            "📊 [DataLoader] Closing stats - Mean: {:.4}, Std: {:.4}",
            closing_mean, closing_std
        );

        if closing_std.abs() < 1e-8 {
            let err_msg = "Closing price standard deviation is too close to zero".to_string();
            println!("❌ [DataLoader] {}", err_msg);
            return Err(TrainingError::DataProcessing(err_msg));
        }

        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        let num_sequences = records.len() - seq_length;
        println!("🔢 [DataLoader] Number of sequences to create: {}", num_sequences);

        for i in 0..num_sequences {
            let seq_slice = feature_matrix.slice(ndarray::s![i..i + seq_length, ..]).to_owned();
            let target = records[i + seq_length].closing;
            if i < 2 {
                println!("📋 [DataLoader] Sample sequence {}: Target = {:.4}", i + 1, target);
                if seq_slice.dim().0 > 0 {
                    let last_step_data = seq_slice.slice(ndarray::s![seq_slice.dim().0 - 1, ..]).to_vec();
                    if last_step_data.len() >= num_features {
                        println!(
                            "📋 [DataLoader] Last step of sequence {}: {:?}",
                            i + 1,
                            &last_step_data[..num_features]
                        );
                    } else {
                        println!("⚠️ [DataLoader] Insufficient data in last step of sequence {}", i + 1);
                    }
                } else {
                    println!("⚠️ [DataLoader] Sequence {} is empty", i + 1);
                }
            }
            sequences.push(seq_slice);
            targets.push(target);
        }

        let feature_stats = FeatureStats {
            feature_means,
            feature_stds,
            feature_names,
            closing_mean,
            closing_std,
        };

        println!(
            "✅ [DataLoader] Created {} sequences of length {}",
            sequences.len(),
            seq_length
        );
        info!("Success: Created {} sequences", sequences.len());
        Ok((sequences, targets, feature_stats))
    }
}

pub fn connect_db(db_url: &str) -> Result<Client, TrainingError> {
    println!("🔧 [Data] Connecting to database...");
    let mut client = Client::connect(db_url, NoTls)?;
    println!("🔍 [Data] Ensuring database tables exist...");
    ensure_tables_exist(&mut client)?;
    println!("✅ [Data] Database connection established and tables verified.");
    Ok(client)
}