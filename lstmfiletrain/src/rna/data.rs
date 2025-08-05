
// projeto: lstmfiletrain
// file: src/rna/data.rs

 

use chrono::{Duration, NaiveDate, Utc, Weekday, Datelike};
use serde::{Serialize, Deserialize};
use rand::{rng, seq::SliceRandom};
use crate::rna::model::{ModelWeights, MultiLayerLSTM};
use crate::LSTMError;
use log::{info, warn};

#[derive(Debug, Serialize, Deserialize)]
pub struct StockData {
    pub date: String,
    pub closing: f32,
    pub opening: f32,
}

pub fn load_all_assets(pg_client: &mut postgres::Client) -> Result<Vec<String>, LSTMError> {
    let rows = pg_client.query("SELECT DISTINCT asset FROM stock_records", &[])?;
    let assets: Vec<String> = rows.into_iter().map(|row| row.get(0)).collect();
    info!("📊 Found {} unique assets: {:?}", assets.len(), assets);
    Ok(assets)
}

pub fn load_data_from_postgres(pg_client: &mut postgres::Client, asset: &str) -> Result<Vec<StockData>, LSTMError> {
    info!("📥 Executing SQL query to load data for asset: {}", asset);
    let rows = pg_client.query(
        "SELECT date, closing, opening FROM stock_records WHERE asset = $1 ORDER BY date ASC",
        &[&asset],
    )?;
    let data: Vec<StockData> = rows
        .into_iter()
        .map(|row| StockData {
            date: row.get(0),
            closing: row.get(1),
            opening: row.get(2),
        })
        .collect();
    if data.is_empty() {
        return Err(LSTMError::InsufficientData {
            asset: asset.to_string(),
            required: 1,
            actual: 0,
        });
    }
    info!("📊 Fetched {} rows for asset {}", data.len(), asset);
    // Log data summary to check for diversity
    let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let unique_closings: std::collections::HashSet<u64> = closing_prices.iter().map(|&x| (x * 1000.0) as u64).collect();
    info!("📊 Unique closing prices for {}: {}", asset, unique_closings.len());
    if unique_closings.len() < closing_prices.len() / 2 {
        warn!("⚠️ Low diversity in closing prices for asset {}: only {} unique values out of {}", asset, unique_closings.len(), closing_prices.len());
    }
    Ok(data)
}

pub fn normalize_data(prices: &[f32]) -> (Vec<f32>, f32, f32) {
    let mean = prices.iter().sum::<f32>() / prices.len().max(1) as f32;
    let variance = prices.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / prices.len().max(1) as f32;
    let std = variance.sqrt().max(1e-8);
    info!("📊 Normalization stats - Mean: {:.4}, Std: {:.4}", mean, std);
    (prices.iter().map(|&x| (x - mean) / std).collect(), mean, std)
}

pub fn calculate_rsi(prices: &[f32], period: usize) -> Vec<f32> {
    let mut rsi = vec![0.0; prices.len()];
    for i in period..prices.len() {
        let slice = &prices[i - period..i];
        let gains = slice.windows(2).map(|w| (w[1] - w[0]).max(0.0)).sum::<f32>();
        let losses = slice.windows(2).map(|w| (w[0] - w[1]).max(0.0)).sum::<f32>();
        if losses < 1e-8 {
            warn!("⚠️ Zero losses detected in RSI calculation for period ending at index {}. Setting RSI to 100.0", i);
            rsi[i] = 100.0; // Handle case where no losses occur
        } else {
            let rs = gains / losses;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    rsi
}

pub fn create_sequences(data: &[StockData], seq_length: usize) -> (Vec<Vec<f32>>, Vec<f32>, f32, f32, f32, f32) {
    let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
    let rsi = calculate_rsi(&closing_prices, 14);
    let (norm_closing, closing_mean, closing_std) = normalize_data(&closing_prices);
    let (norm_opening, opening_mean, opening_std) = normalize_data(&opening_prices);
    let norm_rsi = rsi.iter().map(|&x| (x - 50.0) / 25.0).collect::<Vec<f32>>();

    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for i in 0..data.len().saturating_sub(seq_length) {
        let mut sequence = Vec::with_capacity(seq_length * 5);
        for j in i..i + seq_length {
            sequence.push(norm_closing[j]);
            sequence.push(norm_opening[j]);
            if j >= 4 {
                let ma5 = closing_prices[j - 4..=j].iter().sum::<f32>() / 5.0;
                sequence.push((ma5 - closing_mean) / closing_std);
            } else {
                sequence.push(norm_closing[j]);
            }
            sequence.push(if j > 0 {
                (closing_prices[j] - closing_prices[j - 1]) / closing_std
            } else {
                0.0
            });
            sequence.push(norm_rsi[j]);
        }
        sequences.push(sequence);
        if i + seq_length < data.len() {
            targets.push(norm_closing[i + seq_length]);
        }
    }
    info!("📊 Created {} sequences of length {} (5 features per timestep)", sequences.len(), seq_length);
    (sequences, targets, closing_mean, closing_std, opening_mean, opening_std)
}

pub fn create_batches(sequences: Vec<Vec<f32>>, targets: Vec<f32>, batch_size: usize) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
    let mut rng = rng();
    let mut combined: Vec<(Vec<f32>, f32)> = sequences.into_iter().zip(targets).collect();
    combined.shuffle(&mut rng);
    let batches = combined.chunks(batch_size).map(|chunk| {
        let (seqs, tgts): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
        (seqs, tgts)
    }).collect();
    info!("📊 Created {} batches of size {}", batches.len(), batch_size);
    batches
}

#[allow(dead_code)]
pub fn predict_prices(
    model_weights: ModelWeights,
    data: Vec<StockData>,
    num_predictions: usize,
    verbose: bool,
) -> Result<Vec<(String, f32)>, LSTMError> {
    info!("🧠 Starting prediction process for asset: {}", model_weights.asset);
    if data.len() < model_weights.seq_length {
        return Err(LSTMError::InsufficientData {
            asset: model_weights.asset.clone(),
            required: model_weights.seq_length,
            actual: data.len(),
        });
    }

    info!("🏗️ Building MultiLayerLSTM model from weights...");
    let model = MultiLayerLSTM::from_weights(model_weights.layers, model_weights.w_final, model_weights.b_final)?;
    info!("✅ Model built with {} layers", model.num_layers());

    let closing_prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let opening_prices: Vec<f32> = data.iter().map(|d| d.opening).collect();
    let rsi = calculate_rsi(&closing_prices, 14);
    let (norm_closing, _closing_mean, _closing_std) = normalize_data(&closing_prices);
    let (norm_opening, _opening_mean, _opening_std) = normalize_data(&opening_prices);
    let norm_rsi = rsi.iter().map(|&x| (x - 50.0) / 25.0).collect::<Vec<f32>>();

    // Use the last seq_length data points for the initial sequence
    let start_idx = data.len().saturating_sub(model_weights.seq_length);
    let mut sequence = Vec::with_capacity(model_weights.seq_length * 5);
    for i in start_idx..data.len() {
        sequence.push(norm_closing[i]);
        sequence.push(norm_opening[i]);
        if i >= 4 {
            let ma5 = closing_prices[i - 4..=i].iter().sum::<f32>() / 5.0;
            sequence.push((ma5 - model_weights.closing_mean) / model_weights.closing_std);
        } else {
            sequence.push(norm_closing[i]);
        }
        sequence.push(if i > 0 {
            (closing_prices[i] - closing_prices[i - 1]) / model_weights.closing_std
        } else {
            0.0
        });
        sequence.push(norm_rsi[i]);
    }

    if verbose {
        info!("📈 Last 3 data points used for sequence creation:");
        for i in (data.len().saturating_sub(3))..data.len() {
            info!(
                "  Data[{}]: Date={}, Close={:.2}, Open={:.2}",
                i, data[i].date, data[i].closing, data[i].opening
            );
        }
        info!(
            "🧮 First 6 normalized features of the sequence: {:?}", 
            &sequence[..6.min(sequence.len())]
        );
    }

    let last_date_str = data.last().ok_or_else(|| LSTMError::DataLoadingError("No data available".to_string()))?.date.clone();
    let last_date = NaiveDate::parse_from_str(&last_date_str, "%d.%m.%Y")
        .map_err(|_| LSTMError::DateParseError(format!("Invalid date format: {}", last_date_str)))?;

    let mut predictions = Vec::new();
    let mut current_sequence = sequence;
    let historical_mean = closing_prices.iter().sum::<f32>() / closing_prices.len() as f32;
    let historical_std = (closing_prices.iter().map(|&x| (x - historical_mean).powi(2)).sum::<f32>() / closing_prices.len() as f32).sqrt();

    info!("📅 Starting prediction from date: {}", last_date.format("%d.%m.%Y"));
    for i in 0..num_predictions {
        let norm_pred = model.forward(&current_sequence, false);
        let price = (norm_pred * model_weights.closing_std + model_weights.closing_mean).max(0.0);
        let max_reasonable_price = historical_mean + 3.0 * historical_std;
        let price = price.min(max_reasonable_price);
        let date = next_trading_day(last_date, i as i64);
        predictions.push((date.format("%d.%m.%Y").to_string(), price));

        if verbose {
            info!("🔮 Prediction {}/{}: Date: {}, Price: R$ {:.2}", i + 1, num_predictions, date.format("%d.%m.%Y"), price);
        }

        // Update sequence with predicted value
        current_sequence.drain(0..5); // Remove oldest timestep
        current_sequence.push(norm_pred); // Predicted closing
        current_sequence.push(norm_pred); // Use predicted closing as next opening (simplified)
        // Update MA5 with the new predicted price
        let recent_closings = if i == 0 {
            closing_prices[closing_prices.len().saturating_sub(4)..].to_vec()
        } else {
            let mut recents = closing_prices[closing_prices.len().saturating_sub(4 - i)..].to_vec();
            for j in 0..i.min(4) {
                recents.push(predictions[j].1);
            }
            recents
        };
        let ma5 = recent_closings.iter().sum::<f32>() / recent_closings.len().min(5) as f32;
        current_sequence.push((ma5 - model_weights.closing_mean) / model_weights.closing_std);
        current_sequence.push((price - closing_prices[closing_prices.len().saturating_sub(1 - i)]) / model_weights.closing_std); // Price change
        current_sequence.push(norm_rsi[norm_rsi.len().saturating_sub(1)]); // Reuse last RSI (simplified)
    }

    Ok(predictions)
}

#[allow(dead_code)]
fn next_trading_day(start_date: NaiveDate, offset: i64) -> NaiveDate {
    let mut date = start_date + Duration::days(offset);
    while date.weekday() == Weekday::Sat || date.weekday() == Weekday::Sun {
        date = date + Duration::days(1);
    }
    date
}
