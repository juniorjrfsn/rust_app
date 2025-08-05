// projeto: lstmfilepredict
// file: src/rna/data.rs



pub mod data {
    use serde::{Deserialize, Serialize};
    use postgres::{Client};
    use chrono::{NaiveDate, Duration, Utc};
    use crate::LSTMError;
    use log::info;

    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct StockData {
        pub date: String,
        pub closing: f32,
        pub opening: f32,
    }

    pub fn load_data_from_postgres(client: &mut Client, asset: &str, seq_length: usize, verbose: bool) -> Result<Vec<StockData>, LSTMError> {
        info!("  📥 Executing SQL query to load data for asset: {}", asset);
        let query = "SELECT date, closing, opening FROM stock_records WHERE asset = $1 ORDER BY date ASC";
        let rows = client.query(query, &[&asset])?;

        let mut data = Vec::with_capacity(rows.len());
        info!("  📊 Fetched {} rows from database", rows.len());
        for (i, row) in rows.iter().enumerate() {
            data.push(StockData {
                date: row.get(0),
                closing: row.get(1),
                opening: row.get(2),
            });
            if verbose && (i < 3 || i >= rows.len().saturating_sub(3)) {
                info!("    Row {}: Date={}, Close={}, Open={}", i + 1, row.get::<_, String>(0), row.get::<_, f32>(1), row.get::<_, f32>(2));
            }
        }
        info!("  ✅ Successfully loaded {} data points for {}", data.len(), asset);

        if data.len() < seq_length {
            Err(LSTMError::InsufficientData {
                asset: asset.to_string(),
                required: seq_length,
                actual: data.len(),
            })
        } else {
            Ok(data)
        }
    }

    pub fn normalize_data(data: &[f32], mean: f32, std: f32) -> Vec<f32> {
        let std_safe = std.max(1e-8);
        data.iter().map(|&x| (x - mean) / std_safe).collect()
    }

    pub fn create_prediction_sequence(
        data: &[StockData],
        seq_length: usize,
        closing_mean: f32,
        closing_std: f32,
        opening_mean: f32,
        opening_std: f32,
    ) -> Result<Vec<f32>, LSTMError> {
        if data.len() < seq_length {
            return Err(LSTMError::InsufficientData {
                asset: data.first().map(|d| d.date.clone()).unwrap_or("Unknown".to_string()),
                required: seq_length,
                actual: data.len(),
            });
        }

        let start_index = data.len() - seq_length;
        let closing_prices: Vec<f32> = data[start_index..].iter().map(|d| d.closing).collect();
        let opening_prices: Vec<f32> = data[start_index..].iter().map(|d| d.opening).collect();

        let norm_closing = normalize_data(&closing_prices, closing_mean, closing_std);
        let norm_opening = normalize_data(&opening_prices, opening_mean, opening_std);

        let mut sequence = Vec::new();
        for i in 0..seq_length {
            sequence.push(norm_closing[i]);
            sequence.push(norm_opening[i]);
            if i >= 4 {
                let ma5 = closing_prices[(i - 4)..=i].iter().sum::<f32>() / 5.0;
                sequence.push((ma5 - closing_mean) / closing_std.max(1e-8));
            } else {
                sequence.push(norm_closing[i]);
            }
            if i > 0 {
                sequence.push((closing_prices[i] - closing_prices[i - 1]) / closing_std.max(1e-8));
            } else {
                sequence.push(0.0);
            }
        }

        if data.len() >= 3 && seq_length >= 3 {
            info!("  📈 Last 3 data points used for sequence creation:");
            for i in (seq_length - 3)..seq_length {
                info!("    Data[{}]: Date={}, Close={}, Open={}", start_index + i, data[start_index + i].date, data[start_index + i].closing, data[start_index + i].opening);
            }
            info!("  🧮 First 6 normalized features of the sequence: {:?}", &sequence[..6.min(sequence.len())]);
        }

        Ok(sequence)
    }

    pub fn predict_prices(
        model_weights: crate::rna::model::model::ModelWeights,
        data: Vec<StockData>,
        num_predictions: usize,
        verbose: bool,
    ) -> Result<Vec<(String, f32)>, LSTMError> {
        info!("  🧠 Starting prediction process...");
        let seq_length = model_weights.seq_length;
        let closing_mean = model_weights.closing_mean;
        let closing_std = model_weights.closing_std;
        let opening_mean = model_weights.opening_mean;
        let opening_std = model_weights.opening_std;

        info!("    Model Parameters - Seq Len: {}, Close Mean: {:.4}, Close Std: {:.4}", seq_length, closing_mean, closing_std);

        let model = crate::rna::model::model::MultiLayerLSTM::from_weights(&model_weights);
        let initial_sequence = create_prediction_sequence(&data, seq_length, closing_mean, closing_std, opening_mean, opening_std)?;

        // Use current date as the starting point for predictions
        let current_date = Utc::now().date_naive() + Duration::hours(4); // Adjust to -04 time zone
        info!("  📅 Starting prediction from current date: {}", current_date.format("%d.%m.%Y"));

        let predictions = model.predict_future(initial_sequence, num_predictions, closing_mean, closing_std)?;
        let mut result = Vec::new();
        for (i, pred) in predictions.iter().enumerate() {
            let pred_date = current_date + Duration::days((i + 1) as i64);
            let pred_date_str = pred_date.format("%d.%m.%Y").to_string();
            result.push((pred_date_str.clone(), *pred));
            if verbose {
                println!("  🔮 Prediction {}/{}: Date: {}, Price: R$ {:.2}", i + 1, num_predictions, pred_date_str, pred);
            }
        }

        Ok(result)
    }
}