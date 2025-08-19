// projeto: lstm_cnn_predictor
// file: src/main.rs

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use log::{info, warn, error};
use env_logger;
use thiserror::Error;
use chrono::{DateTime, Utc};
use colored::*;

#[derive(Error, Debug)]
enum PredictorError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("TOML error: {0}")]
    TomlError(#[from] toml::ser::Error),
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

// Estrutura aprimorada para previsões
#[derive(Debug, Serialize, Deserialize, Clone)]
struct PredictionResult {
    asset_symbol: String,
    asset_name: String,
    prediction_date: String,
    last_known_date: String,
    last_known_price: f32,
    predicted_price: f32,
    price_change_absolute: f32,
    price_change_percent: f32,
    confidence_score: f32,
    risk_assessment: String,
    recommendation: String,
    volatility_indicator: f32,
    model_accuracy: f32,
    data_points_used: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    model_type: String,
    version: String,
    training_accuracy: f32,
    validation_accuracy: f32,
    training_date: String,
    prediction_date: String,
    parameters: ModelParameters,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelParameters {
    input_size: usize,
    hidden_size: usize,
    sequence_length: usize,
    learning_rate: f32,
    epochs: usize,
    architecture: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    accuracy_grade: String,
    reliability_score: f32,
    confidence_level: String,
    total_predictions: usize,
    successful_predictions: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionSummary {
    total_assets_analyzed: usize,
    successful_predictions: usize,
    failed_predictions: usize,
    average_confidence: f32,
    bullish_predictions: usize,
    bearish_predictions: usize,
    neutral_predictions: usize,
    high_confidence_predictions: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionOutput {
    metadata: PredictionMetadata,
    model_info: ModelInfo,
    summary: PredictionSummary,
    predictions: Vec<PredictionResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionMetadata {
    generated_at: String,
    generated_by: String,
    format_version: String,
    data_source: String,
    export_formats: Vec<String>,
}

// Estrutura principal do preditor
struct LSTMCNNPredictor {
    weights: ModelWeights,
    normalization_stats: HashMap<String, (f32, f32)>, // (min, max) for each feature
}

impl LSTMCNNPredictor {
    fn from_weights_file(weights_path: &str) -> Result<Self, PredictorError> {
        info!("🔄 Loading model weights from: {}", weights_path);
        
        if !Path::new(weights_path).exists() {
            return Err(PredictorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Weights file not found: {}", weights_path)
            )));
        }
        
        let json_content = fs::read_to_string(weights_path)?;
        let weights: ModelWeights = serde_json::from_str(&json_content)?;
        
        info!("✅ Model weights loaded successfully:");
        info!("   📊 Training accuracy: {:.2}%", weights.training_accuracy * 100.0);
        info!("   📈 Validation accuracy: {:.2}%", weights.validation_accuracy * 100.0);
        info!("   📅 Trained: {}", weights.training_timestamp);
        info!("   🔧 Input size: {}", weights.input_size);
        info!("   🧠 Hidden size: {}", weights.hidden_size);
        info!("   📏 Sequence length: {}", weights.sequence_length);
        
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
        let features: Vec<Vec<f32>> = records.iter().map(|r| {
            vec![r.closing, r.opening, r.high, r.low, r.volume, r.variation]
        }).collect();

        for (col, &feature_name) in feature_names.iter().enumerate() {
            let column: Vec<f32> = features.iter().map(|row| row[col]).collect();
            let min_val = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            self.normalization_stats.insert(feature_name.to_string(), (min_val, max_val));
        }

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

    // Calculate volatility from historical data
    fn calculate_volatility(&self, records: &[StockRecord]) -> f32 {
        if records.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f32> = records.windows(2)
            .map(|w| (w[1].closing / w[0].closing).ln())
            .collect();

        let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f32>() / (returns.len() - 1) as f32;

        (variance * 252.0).sqrt() // Annualized volatility
    }

    // Generate investment recommendation
    fn get_recommendation(&self, change_percent: f32, confidence: f32, volatility: f32) -> (String, String) {
        let risk_level = if volatility > 0.4 {
            "ALTO"
        } else if volatility > 0.25 {
            "MÉDIO"
        } else {
            "BAIXO"
        };

        let recommendation = match (change_percent, confidence) {
            (change, conf) if change > 10.0 && conf > 0.7 => "COMPRA FORTE",
            (change, conf) if change > 5.0 && conf > 0.6 => "COMPRA",
            (change, conf) if change > 0.0 && conf > 0.5 => "COMPRA LEVE",
            (change, conf) if change < -10.0 && conf > 0.7 => "VENDA FORTE",
            (change, conf) if change < -5.0 && conf > 0.6 => "VENDA",
            (change, conf) if change < 0.0 && conf > 0.5 => "VENDA LEVE",
            _ => "MANTER",
        };

        (recommendation.to_string(), risk_level.to_string())
    }

    // Extract asset symbol from full name
    fn extract_asset_symbol(&self, full_name: &str) -> String {
        full_name.split_whitespace()
            .next()
            .unwrap_or(full_name)
            .to_string()
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

        // Calculate metrics
        let last_record = records.last().unwrap();
        let last_price = last_record.closing;
        let price_change_absolute = predicted_price - last_price;
        let price_change_percent = (price_change_absolute / last_price) * 100.0;
        let volatility = self.calculate_volatility(records);
        
        // Adjust confidence based on model accuracy and volatility
        let base_confidence = self.weights.validation_accuracy.max(0.1).min(0.9);
        let volatility_penalty = (volatility * 0.5).min(0.3);
        let confidence_score = (base_confidence - volatility_penalty).max(0.1).min(0.9);

        let (recommendation, risk_assessment) = self.get_recommendation(
            price_change_percent, 
            confidence_score, 
            volatility
        );

        let asset_symbol = self.extract_asset_symbol(asset_name);

        info!("🎯 Prediction for {}: R${:.2} -> R${:.2} ({:+.2}%)", 
            asset_symbol, last_price, predicted_price, price_change_percent);

        Ok(PredictionResult {
            asset_symbol,
            asset_name: asset_name.to_string(),
            prediction_date: Utc::now().format("%Y-%m-%d").to_string(),
            last_known_date: last_record.date.clone(),
            last_known_price: last_price,
            predicted_price,
            price_change_absolute,
            price_change_percent,
            confidence_score,
            risk_assessment,
            recommendation,
            volatility_indicator: volatility,
            model_accuracy: self.weights.validation_accuracy,
            data_points_used: records.len(),
        })
    }
}

// Utility functions
fn load_consolidated_data(file_path: &str) -> Result<ConsolidatedData, PredictorError> {
    info!("📂 Loading consolidated data from: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(PredictorError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {}", file_path)
        )));
    }
    
    let json_content = fs::read_to_string(file_path)?;
    let data: ConsolidatedData = serde_json::from_str(&json_content)?;
    
    info!("✅ Successfully loaded data:");
    info!("   📈 Total assets: {}", data.total_assets);
    info!("   📊 Total records: {}", data.total_records);
    
    Ok(data)
}

fn save_predictions_json(predictions: &PredictionOutput, output_path: &str) -> Result<(), PredictorError> {
    info!("💾 Saving JSON predictions to: {}", output_path);
    
    let json_data = serde_json::to_string_pretty(predictions)?;
    fs::write(output_path, json_data)?;
    
    info!("✅ JSON predictions saved successfully");
    Ok(())
}

fn save_predictions_toml(predictions: &PredictionOutput, output_path: &str) -> Result<(), PredictorError> {
    info!("💾 Saving TOML predictions to: {}", output_path);
    
    let toml_data = toml::to_string_pretty(predictions)?;
    fs::write(output_path, toml_data)?;
    
    info!("✅ TOML predictions saved successfully");
    Ok(())
}

fn calculate_summary_stats(predictions: &[PredictionResult]) -> PredictionSummary {
    let total = predictions.len();
    let average_confidence = if total > 0 {
        predictions.iter().map(|p| p.confidence_score).sum::<f32>() / total as f32
    } else {
        0.0
    };

    let bullish = predictions.iter().filter(|p| p.price_change_percent > 1.0).count();
    let bearish = predictions.iter().filter(|p| p.price_change_percent < -1.0).count();
    let neutral = total - bullish - bearish;
    let high_confidence = predictions.iter().filter(|p| p.confidence_score > 0.7).count();

    PredictionSummary {
        total_assets_analyzed: total,
        successful_predictions: total,
        failed_predictions: 0,
        average_confidence,
        bullish_predictions: bullish,
        bearish_predictions: bearish,
        neutral_predictions: neutral,
        high_confidence_predictions: high_confidence,
    }
}

fn get_accuracy_grade(accuracy: f32) -> String {
    match accuracy {
        a if a >= 0.9 => "A+".to_string(),
        a if a >= 0.8 => "A".to_string(),
        a if a >= 0.7 => "B+".to_string(),
        a if a >= 0.6 => "B".to_string(),
        a if a >= 0.5 => "C+".to_string(),
        a if a >= 0.4 => "C".to_string(),
        a if a >= 0.3 => "D+".to_string(),
        a if a >= 0.2 => "D".to_string(),
        _ => "F".to_string(),
    }
}

fn get_confidence_level(accuracy: f32) -> String {
    match accuracy {
        a if a >= 0.8 => "MUITO ALTA".to_string(),
        a if a >= 0.6 => "ALTA".to_string(),
        a if a >= 0.4 => "MÉDIA".to_string(),
        a if a >= 0.2 => "BAIXA".to_string(),
        _ => "MUITO BAIXA".to_string(),
    }
}

fn display_predictions_table(predictions: &[PredictionResult]) {
    println!("\n{}", "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║                                    📊 RELATÓRIO DE PREVISÕES                                      ║".cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝".cyan());
    
    println!("\n{:<8} {:<12} {:<12} {:<12} {:<8} {:<12} {:<8}", 
        "ATIVO".bold().yellow(),
        "ATUAL".bold().green(),
        "PREVISTO".bold().blue(),
        "VARIAÇÃO".bold().magenta(),
        "CONF.".bold().cyan(),
        "RECOMENDAÇÃO".bold().red(),
        "RISCO".bold().white()
    );
    
    println!("{}", "─".repeat(88));
    
    for pred in predictions.iter().take(15) {
        let change_color = if pred.price_change_percent > 0.0 { "green" } else { "red" };
        let conf_color = if pred.confidence_score > 0.7 { "green" } else if pred.confidence_score > 0.4 { "yellow" } else { "red" };
        
        println!("{:<8} {:<12.2} {:<12.2} {:<12} {:<8} {:<12} {:<8}",
            pred.asset_symbol.bright_white(),
            format!("R${:.2}", pred.last_known_price),
            format!("R${:.2}", pred.predicted_price),
            format!("{:+.2}%", pred.price_change_percent).color(change_color),
            format!("{:.1}%", pred.confidence_score * 100.0).color(conf_color),
            pred.recommendation,
            pred.risk_assessment
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("{}", "🔮 LSTM+CNN Stock Price Predictor v2.0".bright_cyan().bold());
    println!("{}", "═══════════════════════════════════════════════".cyan());
    
    // Configuration parameters
    let weights_path = "../../dados/consolidado/lstm_cnn_weights.json";
    let data_path = "../../dados/consolidado/consolidated_stock_data.json";
    let json_output_path = "../../dados/consolidado/stock_predictions.json";
    let toml_output_path = "../../dados/consolidado/stock_predictions.toml";
    
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
    
    println!("\n{}", "🔄 Processando previsões...".yellow());
    
    for asset_name in &consolidated_data.assets {
        if let Some(asset_records) = consolidated_data.data.get(asset_name) {
            match predictor.predict_asset(asset_name, asset_records) {
                Ok(prediction) => {
                    predictions.push(prediction);
                    successful_predictions += 1;
                }
                Err(e) => {
                    warn!("⚠️ Failed to predict for {}: {}", asset_name, e);
                }
            }
        }
    }
    
    if predictions.is_empty() {
        error!("❌ No predictions were generated");
        return Ok(());
    }
    
    // Sort predictions by confidence
    predictions.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
    
    // Calculate summary statistics
    let summary = calculate_summary_stats(&predictions);
    
    // Prepare output structure
    let prediction_output = PredictionOutput {
        metadata: PredictionMetadata {
            generated_at: Utc::now().to_rfc3339(),
            generated_by: "LSTM+CNN Stock Predictor v2.0".to_string(),
            format_version: "2.0".to_string(),
            data_source: data_path.to_string(),
            export_formats: vec!["JSON".to_string(), "TOML".to_string()],
        },
        model_info: ModelInfo {
            model_type: "LSTM+CNN Hybrid".to_string(),
            version: "1.0".to_string(),
            training_accuracy: predictor.weights.training_accuracy,
            validation_accuracy: predictor.weights.validation_accuracy,
            training_date: predictor.weights.training_timestamp.clone(),
            prediction_date: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            parameters: ModelParameters {
                input_size: predictor.weights.input_size,
                hidden_size: predictor.weights.hidden_size,
                sequence_length: predictor.weights.sequence_length,
                learning_rate: predictor.weights.learning_rate,
                epochs: predictor.weights.epochs,
                architecture: "LSTM + CNN + Dense".to_string(),
            },
            performance_metrics: PerformanceMetrics {
                accuracy_grade: get_accuracy_grade(predictor.weights.validation_accuracy),
                reliability_score: predictor.weights.validation_accuracy,
                confidence_level: get_confidence_level(predictor.weights.validation_accuracy),
                total_predictions: predictions.len(),
                successful_predictions,
            },
        },
        summary,
        predictions,
    };
    
    // Save predictions in both formats
    save_predictions_json(&prediction_output, json_output_path)?;
    save_predictions_toml(&prediction_output, toml_output_path)?;
    
    // Display beautiful console output
    println!("\n{}", "✅ PROCESSAMENTO CONCLUÍDO".bright_green().bold());
    println!("{}", "═══════════════════════════".green());
    
    // Model performance summary
    println!("\n{}", "📈 PERFORMANCE DO MODELO".bright_blue().bold());
    println!("   Acurácia de Treinamento:  {:.2}%", prediction_output.model_info.training_accuracy * 100.0);
    println!("   Acurácia de Validação:    {:.2}%", prediction_output.model_info.validation_accuracy * 100.0);
    println!("   Nota de Confiabilidade:   {}", prediction_output.model_info.performance_metrics.accuracy_grade);
    println!("   Nível de Confiança:       {}", prediction_output.model_info.performance_metrics.confidence_level);
    
    // Summary statistics
    println!("\n{}", "📊 ESTATÍSTICAS GERAIS".bright_magenta().bold());
    println!("   Total de Ativos:          {}", prediction_output.summary.total_assets_analyzed);
    println!("   Previsões Bem-sucedidas:  {}", prediction_output.summary.successful_predictions);
    println!("   Confiança Média:          {:.1}%", prediction_output.summary.average_confidence * 100.0);
    println!("   Previsões Otimistas:      {} ({:.1}%)", 
        prediction_output.summary.bullish_predictions,
        (prediction_output.summary.bullish_predictions as f32 / prediction_output.summary.total_assets_analyzed as f32) * 100.0
    );
    println!("   Previsões Pessimistas:    {} ({:.1}%)", 
        prediction_output.summary.bearish_predictions,
        (prediction_output.summary.bearish_predictions as f32 / prediction_output.summary.total_assets_analyzed as f32) * 100.0
    );
    println!("   Previsões Neutras:        {} ({:.1}%)", 
        prediction_output.summary.neutral_predictions,
        (prediction_output.summary.neutral_predictions as f32 / prediction_output.summary.total_assets_analyzed as f32) * 100.0
    );
    println!("   Alta Confiança (>70%):    {}", prediction_output.summary.high_confidence_predictions);
    
    // Display predictions table
    display_predictions_table(&prediction_output.predictions);
    
    // Top recommendations
    println!("\n{}", "🏆 TOP 5 RECOMENDAÇÕES".bright_yellow().bold());
    println!("{}", "═══════════════════════".yellow());
    
    for (i, pred) in prediction_output.predictions.iter().take(5).enumerate() {
        let trend_icon = if pred.price_change_percent > 5.0 {
            "🚀"
        } else if pred.price_change_percent > 0.0 {
            "📈"
        } else if pred.price_change_percent < -5.0 {
            "📉"
        } else {
            "➡️"
        };
        
        println!("{}. {} {} - {} -> {} ({:+.2}%)", 
            i + 1,
            trend_icon,
            pred.asset_symbol.bright_white().bold(),
            format!("R${:.2}", pred.last_known_price).green(),
            format!("R${:.2}", pred.predicted_price).blue(),
            pred.price_change_percent
        );
        println!("      Recomendação: {} | Risco: {} | Confiança: {:.1}%",
            pred.recommendation.bright_cyan(),
            pred.risk_assessment.bright_yellow(),
            pred.confidence_score * 100.0
        );
        println!();
    }
    
    // Risk analysis
    let high_risk_count = prediction_output.predictions.iter()
        .filter(|p| p.risk_assessment == "ALTO")
        .count();
    let medium_risk_count = prediction_output.predictions.iter()
        .filter(|p| p.risk_assessment == "MÉDIO")
        .count();
    let low_risk_count = prediction_output.predictions.iter()
        .filter(|p| p.risk_assessment == "BAIXO")
        .count();
    
    println!("{}", "⚠️  ANÁLISE DE RISCO".bright_red().bold());
    println!("{}", "═══════════════════".red());
    println!("   🔴 Alto Risco:     {} ativos", high_risk_count);
    println!("   🟡 Médio Risco:    {} ativos", medium_risk_count);
    println!("   🟢 Baixo Risco:    {} ativos", low_risk_count);
    
    // Export information
    println!("\n{}", "💾 ARQUIVOS GERADOS".bright_green().bold());
    println!("{}", "═══════════════════".green());
    println!("   📄 JSON: {}", json_output_path);
    println!("   📄 TOML: {}", toml_output_path);
    
    // Warning about low accuracy
    if prediction_output.model_info.validation_accuracy < 0.5 {
        println!("\n{}", "⚠️  AVISO IMPORTANTE".bright_red().bold());
        println!("{}", "════════════════════".red());
        println!("   A acurácia do modelo ({:.1}%) está abaixo do recomendado.", 
            prediction_output.model_info.validation_accuracy * 100.0);
        println!("   Use estas previsões apenas como referência.");
        println!("   Considere retreinar o modelo com mais dados ou ajustar os parâmetros.");
    }
    
    println!("\n{}", "🎯 Previsões concluídas com sucesso!".bright_green().bold());
    
    Ok(())
}

// Para executar:
// cargo run
 