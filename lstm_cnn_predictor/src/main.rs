// projeto: lstm_cnn_predictor
// file: src/main.rs

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Axis};
use log::{info, warn, error};
use env_logger;
use thiserror::Error;

#[derive(Error, Debug)]
enum PredictorError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Data processing error: {msg}")]
    DataError { msg: String },
    #[error("Model loading error: {msg}")]
    ModelError { msg: String },
}

// Estruturas para carregar os dados JSON (copiadas do projeto original)
#[derive(Debug, Serialize, Deserialize, Clone)]
struct StockRecord {
    asset: String,
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConsolidatedData {
    total_assets: usize,
    total_records: usize,
    assets: Vec<String>,
    export_timestamp: String,
    data: HashMap<String, Vec<StockRecord>>,
}

// Estruturas dos pesos do modelo (copiadas do projeto original)
#[derive(Debug, Serialize, Deserialize)]
struct LSTMWeights {
    w_input: Vec<Vec<f32>>,
    w_forget: Vec<Vec<f32>>,
    w_output: Vec<Vec<f32>>,
    w_candidate: Vec<Vec<f32>>,
    
    b_input: Vec<f32>,
    b_forget: Vec<f32>,
    b_output: Vec<f32>,
    b_candidate: Vec<f32>,
    
    u_input: Vec<Vec<f32>>,
    u_forget: Vec<Vec<f32>>,
    u_output: Vec<Vec<f32>>,
    u_candidate: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CNNWeights {
    conv_filters: Vec<Vec<Vec<f32>>>,
    conv_biases: Vec<f32>,
    pool_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DenseWeights {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelWeights {
    lstm: LSTMWeights,
    cnn: CNNWeights,
    dense: DenseWeights,
    input_size: usize,
    hidden_size: usize,
    sequence_length: usize,
    num_classes: usize,
    learning_rate: f32,
    epochs: usize,
    training_accuracy: f32,
    validation_accuracy: f32,
    training_timestamp: String,
}

// Estrutura para previsões
#[derive(Debug, Serialize, Deserialize)]
struct PredictionResult {
    asset: String,
    prediction_date: String,
    predicted_price: f32,
    confidence: f32,
    last_known_price: f32,
    price_change_percent: f32,
    input_sequence: Vec<Vec<f32>>,
    model_accuracy: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionOutput {
    model_info: ModelInfo,
    predictions: Vec<PredictionResult>,
    generation_timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    training_accuracy: f32,
    validation_accuracy: f32,
    training_timestamp: String,
    model_parameters: ModelParameters,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelParameters {
    input_size: usize,
    hidden_size: usize,
    sequence_length: usize,
    learning_rate: f32,
    epochs: usize,
}

// Estrutura principal do preditor
struct LSTMCNNPredictor {
    weights: ModelWeights,
    normalization_stats: HashMap<String, (f32, f32)>, // (min, max) for each feature
}

impl LSTMCNNPredictor {
    fn from_weights_file(weights_path: &str) -> Result<Self, PredictorError> {
        info!("Loading model weights from: {}", weights_path);
        
        if !Path::new(weights_path).exists() {
            return Err(PredictorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Weights file not found: {}", weights_path)
            )));
        }
        
        let json_content = fs::read_to_string(weights_path)?;
        let weights: ModelWeights = serde_json::from_str(&json_content)?;
        
        info!("✅ Model weights loaded successfully:");
        info!("  Training accuracy: {:.2}%", weights.training_accuracy * 100.0);
        info!("  Validation accuracy: {:.2}%", weights.validation_accuracy * 100.0);
        info!("  Trained: {}", weights.training_timestamp);
        info!("  Input size: {}", weights.input_size);
        info!("  Hidden size: {}", weights.hidden_size);
        info!("  Sequence length: {}", weights.sequence_length);
        
        Ok(Self {
            weights,
            normalization_stats: HashMap::new(),
        })
    }

    // Activation functions (copiadas do modelo original)
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    // LSTM forward pass
    fn lstm_forward(&self, input_sequence: &Array2<f32>) -> Array1<f32> {
        let mut hidden_state = Array1::zeros(self.weights.hidden_size);
        let mut cell_state = Array1::zeros(self.weights.hidden_size);

        // Process each time step
        for t in 0..input_sequence.nrows() {
            let input = input_sequence.row(t);
            
            // Input gate
            let input_gate: Array1<f32> = Array1::from_vec(
                (0..self.weights.hidden_size)
                    .map(|i| {
                        let mut sum = self.weights.lstm.b_input[i];
                        for j in 0..self.weights.input_size {
                            sum += self.weights.lstm.w_input[i][j] * input[j];
                        }
                        for j in 0..self.weights.hidden_size {
                            sum += self.weights.lstm.u_input[i][j] * hidden_state[j];
                        }
                        Self::sigmoid(sum)
                    })
                    .collect()
            );

            // Forget gate
            let forget_gate: Array1<f32> = Array1::from_vec(
                (0..self.weights.hidden_size)
                    .map(|i| {
                        let mut sum = self.weights.lstm.b_forget[i];
                        for j in 0..self.weights.input_size {
                            sum += self.weights.lstm.w_forget[i][j] * input[j];
                        }
                        for j in 0..self.weights.hidden_size {
                            sum += self.weights.lstm.u_forget[i][j] * hidden_state[j];
                        }
                        Self::sigmoid(sum)
                    })
                    .collect()
            );

            // Output gate
            let output_gate: Array1<f32> = Array1::from_vec(
                (0..self.weights.hidden_size)
                    .map(|i| {
                        let mut sum = self.weights.lstm.b_output[i];
                        for j in 0..self.weights.input_size {
                            sum += self.weights.lstm.w_output[i][j] * input[j];
                        }
                        for j in 0..self.weights.hidden_size {
                            sum += self.weights.lstm.u_output[i][j] * hidden_state[j];
                        }
                        Self::sigmoid(sum)
                    })
                    .collect()
            );

            // Candidate values
            let candidate: Array1<f32> = Array1::from_vec(
                (0..self.weights.hidden_size)
                    .map(|i| {
                        let mut sum = self.weights.lstm.b_candidate[i];
                        for j in 0..self.weights.input_size {
                            sum += self.weights.lstm.w_candidate[i][j] * input[j];
                        }
                        for j in 0..self.weights.hidden_size {
                            sum += self.weights.lstm.u_candidate[i][j] * hidden_state[j];
                        }
                        Self::tanh(sum)
                    })
                    .collect()
            );

            // Update cell state and hidden state
            cell_state = &forget_gate * &cell_state + &input_gate * &candidate;
            hidden_state = &output_gate * &cell_state.map(|x| Self::tanh(*x));
        }

        hidden_state
    }

    // CNN forward pass
    fn cnn_forward(&self, input: &Array2<f32>) -> Array1<f32> {
        let mut features = Vec::new();

        // Apply each filter
        for (filter_idx, filter) in self.weights.cnn.conv_filters.iter().enumerate() {
            let mut conv_output = Vec::new();
            
            // 1D convolution over time dimension
            for i in 0..=(input.nrows() - filter.len()) {
                let mut sum = self.weights.cnn.conv_biases[filter_idx];
                for j in 0..filter.len() {
                    for k in 0..input.ncols() {
                        sum += filter[j][0] * input[[i + j, k]];
                    }
                }
                conv_output.push(Self::relu(sum));
            }
            
            // Max pooling
            if !conv_output.is_empty() {
                let pooled = conv_output
                    .chunks(self.weights.cnn.pool_size)
                    .map(|chunk| chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                features.push(pooled);
            }
        }

        Array1::from_vec(features)
    }

    // Forward pass combining LSTM and CNN
    fn forward(&self, input: &Array2<f32>) -> f32 {
        // LSTM processing
        let lstm_output = self.lstm_forward(input);
        
        // CNN processing
        let cnn_output = self.cnn_forward(input);
        
        // Combine features
        let combined_features = Array1::from_iter(
            lstm_output.iter().chain(cnn_output.iter()).cloned()
        );
        
        // Dense layer
        let mut output = self.weights.dense.biases[0];
        for (i, &feature) in combined_features.iter().enumerate() {
            if i < self.weights.dense.weights[0].len() {
                output += self.weights.dense.weights[0][i] * feature;
            }
        }
        
        output
    }

    // Calculate normalization statistics from historical data
    fn calculate_normalization_stats(&mut self, records: &[StockRecord]) -> Result<(), PredictorError> {
        if records.is_empty() {
            return Err(PredictorError::DataError { msg: "No records provided for normalization".to_string() });
        }

        let feature_names = ["closing", "opening", "high", "low", "volume", "variation"];
        let mut features: Vec<Vec<f32>> = records.iter().map(|r| {
            vec![r.closing, r.opening, r.high, r.low, r.volume, r.variation]
        }).collect();

        for (col, &feature_name) in feature_names.iter().enumerate() {
            let column: Vec<f32> = features.iter().map(|row| row[col]).collect();
            let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            self.normalization_stats.insert(feature_name.to_string(), (min_val, max_val));
        }

        info!("Normalization statistics calculated for {} features", feature_names.len());
        Ok(())
    }

    // Normalize a single sequence
    fn normalize_sequence(&self, features: Vec<Vec<f32>>) -> Result<Array2<f32>, PredictorError> {
        let feature_names = ["closing", "opening", "high", "low", "volume", "variation"];
        let mut normalized_features = features.clone();

        for (col, &feature_name) in feature_names.iter().enumerate() {
            if let Some(&(min_val, max_val)) = self.normalization_stats.get(feature_name) {
                let range = max_val - min_val;
                if range > 0.0 {
                    for row in normalized_features.iter_mut() {
                        row[col] = (row[col] - min_val) / range;
                    }
                }
            }
        }

        let sequence_length = normalized_features.len();
        let feature_count = normalized_features[0].len();
        let flat_data: Vec<f32> = normalized_features.into_iter().flatten().collect();

        Array2::from_shape_vec((sequence_length, feature_count), flat_data)
            .map_err(|e| PredictorError::DataError { msg: format!("Failed to create normalized array: {}", e) })
    }

    // Denormalize prediction (convert back to original scale)
    fn denormalize_prediction(&self, normalized_value: f32) -> f32 {
        if let Some(&(min_val, max_val)) = self.normalization_stats.get("closing") {
            let range = max_val - min_val;
            if range > 0.0 {
                return normalized_value * range + min_val;
            }
        }
        normalized_value
    }

    // Make prediction for a single asset
    fn predict_asset(&mut self, asset_name: &str, records: &[StockRecord]) -> Result<PredictionResult, PredictorError> {
        if records.len() < self.weights.sequence_length {
            return Err(PredictorError::DataError {
                msg: format!("Not enough data for {}. Need {} records, got {}", 
                    asset_name, self.weights.sequence_length, records.len())
            });
        }

        // Calculate normalization stats from all available data
        self.calculate_normalization_stats(records)?;

        // Get the last sequence for prediction
        let start_idx = records.len() - self.weights.sequence_length;
        let last_sequence: Vec<Vec<f32>> = records[start_idx..].iter().map(|r| {
            vec![r.closing, r.opening, r.high, r.low, r.volume, r.variation]
        }).collect();

        // Normalize the sequence
        let normalized_sequence = self.normalize_sequence(last_sequence.clone())?;

        // Make prediction
        let normalized_prediction = self.forward(&normalized_sequence);
        let predicted_price = self.denormalize_prediction(normalized_prediction);

        // Calculate confidence (simplified as inverse of loss, clamped between 0-1)
        let last_price = records.last().unwrap().closing;
        let price_change_percent = ((predicted_price - last_price) / last_price) * 100.0;
        let confidence = (self.weights.validation_accuracy).max(0.1).min(0.9); // Use model's validation accuracy

        info!("Prediction for {}: {:.2} -> {:.2} ({:+.2}%)", 
            asset_name, last_price, predicted_price, price_change_percent);

        Ok(PredictionResult {
            asset: asset_name.to_string(),
            prediction_date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            predicted_price,
            confidence,
            last_known_price: last_price,
            price_change_percent,
            input_sequence: last_sequence,
            model_accuracy: self.weights.validation_accuracy,
        })
    }
}

// Utility functions
fn load_consolidated_data(file_path: &str) -> Result<ConsolidatedData, PredictorError> {
    info!("Loading consolidated data from: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(PredictorError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {}", file_path)
        )));
    }
    
    let json_content = fs::read_to_string(file_path)?;
    let data: ConsolidatedData = serde_json::from_str(&json_content)?;
    
    info!("✅ Successfully loaded data:");
    info!("  Total assets: {}", data.total_assets);
    info!("  Total records: {}", data.total_records);
    
    Ok(data)
}

fn save_predictions(predictions: &PredictionOutput, output_path: &str) -> Result<(), PredictorError> {
    info!("Saving predictions to: {}", output_path);
    
    let json_data = serde_json::to_string_pretty(predictions)?;
    fs::write(output_path, json_data)?;
    
    info!("✅ Predictions saved successfully");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("🔮 Starting LSTM+CNN Stock Price Prediction");
    
    // Configuration parameters
    let weights_path = "../../dados/consolidado/lstm_cnn_weights.json";
    let data_path = "../../dados/consolidado/consolidated_stock_data.json";
    let output_path = "../../dados/consolidado/stock_predictions.json";
    
    // Load trained model
    let mut predictor = LSTMCNNPredictor::from_weights_file(weights_path)?;
    
    // Load stock data
    let consolidated_data = load_consolidated_data(data_path)?;
    
    if consolidated_data.data.is_empty() {
        warn!("⚠️ No asset data found");
        return Ok(());
    }
    
    // Make predictions for all assets
    let mut predictions = Vec::new();
    let mut successful_predictions = 0;
    
    for asset_name in &consolidated_data.assets {
        if let Some(asset_records) = consolidated_data.data.get(asset_name) {
            match predictor.predict_asset(asset_name, asset_records) {
                Ok(prediction) => {
                    predictions.push(prediction);
                    successful_predictions += 1;
                }
                Err(e) => {
                    warn!("Failed to predict for {}: {}", asset_name, e);
                }
            }
        }
    }
    
    if predictions.is_empty() {
        error!("❌ No predictions were generated");
        return Ok(());
    }
    
    // Prepare output
    let prediction_output = PredictionOutput {
        model_info: ModelInfo {
            training_accuracy: predictor.weights.training_accuracy,
            validation_accuracy: predictor.weights.validation_accuracy,
            training_timestamp: predictor.weights.training_timestamp.clone(),
            model_parameters: ModelParameters {
                input_size: predictor.weights.input_size,
                hidden_size: predictor.weights.hidden_size,
                sequence_length: predictor.weights.sequence_length,
                learning_rate: predictor.weights.learning_rate,
                epochs: predictor.weights.epochs,
            },
        },
        predictions,
        generation_timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    // Save predictions
    save_predictions(&prediction_output, output_path)?;
    
    // Display results
    println!("🎯 Prediction Results:");
    println!("  Total assets processed: {}", consolidated_data.assets.len());
    println!("  Successful predictions: {}", successful_predictions);
    println!("  Model validation accuracy: {:.2}%", predictor.weights.validation_accuracy * 100.0);
    println!("\n📊 Top Predictions:");
    
    // Sort predictions by confidence and show top 5
    let mut sorted_predictions = prediction_output.predictions.clone();
    sorted_predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    for (i, pred) in sorted_predictions.iter().take(5).enumerate() {
        println!("{}. {} - Predicted: R${:.2} (Change: {:+.2}%, Confidence: {:.1}%)",
            i + 1,
            pred.asset,
            pred.predicted_price,
            pred.price_change_percent,
            pred.confidence * 100.0
        );
    }
    
    println!("\n✅ Predictions saved to: {}", output_path);
    
    Ok(())
}

// Para executar:
// cargo run