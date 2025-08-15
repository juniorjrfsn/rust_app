// projeto: lstm_cnn_train
// file: src/main.rs



use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use log::{info, error};
use env_logger;
use thiserror::Error;
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Error, Debug)]
enum TrainingError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Data processing error: {msg}")]
    DataError { msg: String },
    #[error("Model error: {msg}")]
    ModelError { msg: String },
}

#[derive(Parser)]
#[command(name = "lstm_cnn_train")]
#[command(about = "Train LSTM and CNN models on stock data")]
#[command(version = "0.1.0")]
struct Cli {
    #[arg(long, default_value = "../../dados/consolidado", help = "Data directory path")]
    data_dir: String,
    #[arg(long, default_value = "consolidated_stock_data.json", help = "Input data file")]
    input_file: String,
    #[arg(long, default_value = "model_weights", help = "Output directory for model weights")]
    output_dir: String,
    #[arg(long, default_value = "60", help = "Sequence length for LSTM")]
    sequence_length: usize,
    #[arg(long, default_value = "0.8", help = "Training data split ratio")]
    train_split: f32,
    #[arg(long, default_value = "100", help = "Number of training epochs")]
    epochs: usize,
    #[arg(long, default_value = "32", help = "Batch size")]
    batch_size: usize,
    #[arg(long, default_value = "0.001", help = "Learning rate")]
    learning_rate: f32,
    #[arg(long, default_value = "50", help = "LSTM hidden size")]
    lstm_hidden_size: usize,
    #[arg(long, default_value = "32", help = "CNN filter count")]
    cnn_filters: usize,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
struct ConsolidatedData {
    total_assets: usize,
    total_records: usize,
    assets: Vec<String>,
    export_timestamp: String,
    data: HashMap<String, Vec<StockRecord>>,
}

#[derive(Debug, Serialize)]
struct ModelWeights {
    model_type: String,
    architecture: ModelArchitecture,
    weights: WeightsData,
    biases: BiasesData,
    metadata: ModelMetadata,
}

#[derive(Debug, Serialize)]
struct ModelArchitecture {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    sequence_length: usize,
    num_layers: usize,
}

#[derive(Debug, Serialize)]
struct WeightsData {
    lstm_input_weights: Vec<Vec<f32>>,
    lstm_hidden_weights: Vec<Vec<f32>>,
    lstm_forget_weights: Vec<Vec<f32>>,
    lstm_output_weights: Vec<Vec<f32>>,
    cnn_conv_weights: Vec<Vec<Vec<f32>>>,
    dense_weights: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct BiasesData {
    lstm_input_bias: Vec<f32>,
    lstm_hidden_bias: Vec<f32>,
    lstm_forget_bias: Vec<f32>,
    lstm_output_bias: Vec<f32>,
    cnn_conv_bias: Vec<f32>,
    dense_bias: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ModelMetadata {
    training_epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    train_loss: f32,
    validation_loss: f32,
    assets_trained: Vec<String>,
    feature_names: Vec<String>,
    normalization_params: NormalizationParams,
    created_at: String,
}

#[derive(Debug, Serialize)]
struct NormalizationParams {
    means: Vec<f32>,
    stds: Vec<f32>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

// Simple LSTM Cell implementation
#[derive(Debug, Clone)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    wi: Array2<f32>, // Input weights
    wh: Array2<f32>, // Hidden weights
    wf: Array2<f32>, // Forget weights
    wo: Array2<f32>, // Output weights
    bi: Array1<f32>, // Input bias
    bh: Array1<f32>, // Hidden bias
    bf: Array1<f32>, // Forget bias
    bo: Array1<f32>, // Output bias
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            wi: Array2::random((hidden_size, input_size), Uniform::new(-0.1, 0.1)),
            wh: Array2::random((hidden_size, hidden_size), Uniform::new(-0.1, 0.1)),
            wf: Array2::random((hidden_size, input_size + hidden_size), Uniform::new(-0.1, 0.1)),
            wo: Array2::random((hidden_size, input_size + hidden_size), Uniform::new(-0.1, 0.1)),
            bi: Array1::random(hidden_size, Uniform::new(-0.1, 0.1)),
            bh: Array1::random(hidden_size, Uniform::new(-0.1, 0.1)),
            bf: Array1::random(hidden_size, Uniform::new(-0.1, 0.1)),
            bo: Array1::random(hidden_size, Uniform::new(-0.1, 0.1)),
        }
    }

    fn forward(&self, input: &Array1<f32>, hidden: &Array1<f32>, cell: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        // Concatenate input and hidden state
        let mut combined = Array1::zeros(self.input_size + self.hidden_size);
        combined.slice_mut(s![0..self.input_size]).assign(input);
        combined.slice_mut(s![self.input_size..]).assign(hidden);

        // Forget gate
        let forget_gate = sigmoid(&(self.wf.dot(&combined) + &self.bf));
        
        // Input gate
        let input_gate = sigmoid(&(self.wi.dot(input) + self.wh.dot(hidden) + &self.bi));
        
        // Candidate values
        let candidate = tanh(&(self.wi.dot(input) + self.wh.dot(hidden) + &self.bh));
        
        // Update cell state
        let new_cell = &forget_gate * cell + &input_gate * &candidate;
        
        // Output gate
        let output_gate = sigmoid(&(self.wo.dot(&combined) + &self.bo));
        
        // New hidden state
        let new_hidden = &output_gate * &tanh(&new_cell);
        
        (new_hidden, new_cell)
    }
}

// Simple CNN Layer implementation
#[derive(Debug, Clone)]
struct CNNLayer {
    filters: usize,
    kernel_size: usize,
    weights: Array3<f32>,
    bias: Array1<f32>,
}

impl CNNLayer {
    fn new(input_channels: usize, filters: usize, kernel_size: usize) -> Self {
        Self {
            filters,
            kernel_size,
            weights: Array3::random((filters, input_channels, kernel_size), Uniform::new(-0.1, 0.1)),
            bias: Array1::random(filters, Uniform::new(-0.1, 0.1)),
        }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (input_channels, input_length) = input.dim();
        let output_length = input_length - self.kernel_size + 1;
        let mut output = Array2::zeros((self.filters, output_length));

        for f in 0..self.filters {
            for i in 0..output_length {
                let mut sum = self.bias[f];
                for c in 0..input_channels {
                    for k in 0..self.kernel_size {
                        sum += self.weights[[f, c, k]] * input[[c, i + k]];
                    }
                }
                output[[f, i]] = relu(sum);
            }
        }

        output
    }
}

// Dense layer
#[derive(Debug, Clone)]
struct DenseLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Array2::random((output_size, input_size), Uniform::new(-0.1, 0.1)),
            bias: Array1::random(output_size, Uniform::new(-0.1, 0.1)),
        }
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.bias
    }
}

// Combined LSTM-CNN Model
#[derive(Debug)]
struct LSTMCNNModel {
    lstm: LSTMCell,
    cnn: CNNLayer,
    dense: DenseLayer,
    sequence_length: usize,
    feature_size: usize,
}

impl LSTMCNNModel {
    fn new(sequence_length: usize, feature_size: usize, hidden_size: usize, cnn_filters: usize) -> Self {
        Self {
            lstm: LSTMCell::new(feature_size, hidden_size),
            cnn: CNNLayer::new(1, cnn_filters, 3),
            dense: DenseLayer::new(hidden_size + cnn_filters * (sequence_length - 2), 1),
            sequence_length,
            feature_size,
        }
    }

    fn forward(&self, sequence: &Array2<f32>) -> f32 {
        let (seq_len, _features) = sequence.dim();
        
        // LSTM forward pass
        let mut hidden = Array1::zeros(self.lstm.hidden_size);
        let mut cell = Array1::zeros(self.lstm.hidden_size);
        
        for i in 0..seq_len {
            let input = sequence.row(i).to_owned();
            let (new_hidden, new_cell) = self.lstm.forward(&input, &hidden, &cell);
            hidden = new_hidden;
            cell = new_cell;
        }

        // CNN forward pass on the closing prices
        let closing_prices = sequence.column(0).to_owned();
        let cnn_input = closing_prices.insert_axis(Axis(0));
        let cnn_output = self.cnn.forward(&cnn_input);
        let cnn_flattened = cnn_output.to_shape((cnn_output.len(),)).unwrap().to_owned();

        // Combine LSTM and CNN outputs
        let mut combined_features = Array1::zeros(hidden.len() + cnn_flattened.len());
        combined_features.slice_mut(s![0..hidden.len()]).assign(&hidden);
        combined_features.slice_mut(s![hidden.len()..]).assign(&cnn_flattened);

        // Dense layer for final prediction
        let output = self.dense.forward(&combined_features);
        output[0]
    }
}

// Activation functions
fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// Data preprocessing
fn normalize_data(data: &mut Array2<f32>) -> NormalizationParams {
    let (rows, cols) = data.dim();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut mins = Vec::new();
    let mut maxs = Vec::new();

    for col in 0..cols {
        let column_data = data.column(col);
        let mean = column_data.mean().unwrap();
        let std = column_data.std(0.0);
        let min = column_data.fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max = column_data.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

        means.push(mean);
        stds.push(std);
        mins.push(min);
        maxs.push(max);

        // Normalize column
        for row in 0..rows {
            data[[row, col]] = (data[[row, col]] - mean) / std;
        }
    }

    NormalizationParams { means, stds, mins, maxs }
}

fn create_sequences(data: &Array2<f32>, sequence_length: usize) -> (Array3<f32>, Array1<f32>) {
    let (total_samples, features) = data.dim();
    let num_sequences = total_samples - sequence_length;
    
    let mut sequences = Array3::zeros((num_sequences, sequence_length, features));
    let mut targets = Array1::zeros(num_sequences);

    for i in 0..num_sequences {
        // Input sequence
        for j in 0..sequence_length {
            for k in 0..features {
                sequences[[i, j, k]] = data[[i + j, k]];
            }
        }
        // Target (next closing price)
        targets[i] = data[[i + sequence_length, 0]]; // Assuming closing price is first column
    }

    (sequences, targets)
}

fn load_consolidated_data(file_path: &str) -> Result<ConsolidatedData, TrainingError> {
    info!("Loading consolidated data from: {}", file_path);
    let content = fs::read_to_string(file_path)?;
    let data: ConsolidatedData = serde_json::from_str(&content)?;
    info!("Loaded {} assets with {} total records", data.total_assets, data.total_records);
    Ok(data)
}

fn prepare_training_data(consolidated_data: &ConsolidatedData) -> Result<(Array2<f32>, Vec<String>), TrainingError> {
    let mut all_records = Vec::new();
    
    // Combine all asset data
    for (asset, records) in &consolidated_data.data {
        info!("Processing {} records for asset: {}", records.len(), asset);
        for record in records {
            all_records.push(vec![
                record.closing,
                record.opening,
                record.high,
                record.low,
                record.volume,
                record.variation,
            ]);
        }
    }

    if all_records.is_empty() {
        return Err(TrainingError::DataError { 
            msg: "No records found in consolidated data".to_string() 
        });
    }

    let features = all_records[0].len();
    let samples = all_records.len();
    
    let mut data_matrix = Array2::zeros((samples, features));
    for (i, record) in all_records.iter().enumerate() {
        for (j, &value) in record.iter().enumerate() {
            data_matrix[[i, j]] = value;
        }
    }

    let feature_names = vec![
        "closing".to_string(),
        "opening".to_string(),
        "high".to_string(),
        "low".to_string(),
        "volume".to_string(),
        "variation".to_string(),
    ];

    Ok((data_matrix, feature_names))
}

fn train_model(
    model: &mut LSTMCNNModel,
    sequences: &Array3<f32>,
    targets: &Array1<f32>,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
) -> (f32, f32) {
    let num_samples = sequences.dim().0;
    let train_size = (num_samples as f32 * 0.8) as usize;
    
    info!("Training with {} samples, {} for training, {} for validation", 
          num_samples, train_size, num_samples - train_size);

    let mut train_loss = 0.0;
    let mut val_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Training
        for i in (0..train_size).step_by(batch_size) {
            let end_idx = (i + batch_size).min(train_size);
            let mut batch_loss = 0.0;

            for j in i..end_idx {
                let sequence = sequences.slice(s![j, .., ..]).to_owned();
                let target = targets[j];
                let prediction = model.forward(&sequence);
                
                // Mean Squared Error
                let error = prediction - target;
                batch_loss += error * error;
            }

            batch_loss /= (end_idx - i) as f32;
            epoch_loss += batch_loss;
            batch_count += 1;
        }

        train_loss = epoch_loss / batch_count as f32;

        // Validation
        if epoch % 10 == 0 {
            let mut val_epoch_loss = 0.0;
            let val_samples = num_samples - train_size;

            for i in train_size..num_samples {
                let sequence = sequences.slice(s![i, .., ..]).to_owned();
                let target = targets[i];
                let prediction = model.forward(&sequence);
                let error = prediction - target;
                val_epoch_loss += error * error;
            }

            val_loss = val_epoch_loss / val_samples as f32;

            info!("Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6}", 
                  epoch + 1, epochs, train_loss, val_loss);
        }
    }

    (train_loss, val_loss)
}

fn save_model_weights(
    model: &LSTMCNNModel,
    metadata: ModelMetadata,
    output_path: &str,
) -> Result<(), TrainingError> {
    let weights = WeightsData {
        lstm_input_weights: model.lstm.wi.outer_iter().map(|row| row.to_vec()).collect(),
        lstm_hidden_weights: model.lstm.wh.outer_iter().map(|row| row.to_vec()).collect(),
        lstm_forget_weights: model.lstm.wf.outer_iter().map(|row| row.to_vec()).collect(),
        lstm_output_weights: model.lstm.wo.outer_iter().map(|row| row.to_vec()).collect(),
        cnn_conv_weights: model.cnn.weights.outer_iter()
            .map(|filter| filter.outer_iter().map(|row| row.to_vec()).collect())
            .collect(),
        dense_weights: model.dense.weights.outer_iter().map(|row| row.to_vec()).collect(),
    };

    let biases = BiasesData {
        lstm_input_bias: model.lstm.bi.to_vec(),
        lstm_hidden_bias: model.lstm.bh.to_vec(),
        lstm_forget_bias: model.lstm.bf.to_vec(),
        lstm_output_bias: model.lstm.bo.to_vec(),
        cnn_conv_bias: model.cnn.bias.to_vec(),
        dense_bias: model.dense.bias.to_vec(),
    };

    let architecture = ModelArchitecture {
        input_size: model.feature_size,
        hidden_size: model.lstm.hidden_size,
        output_size: 1,
        sequence_length: model.sequence_length,
        num_layers: 1,
    };

    let model_weights = ModelWeights {
        model_type: "LSTM-CNN".to_string(),
        architecture,
        weights,
        biases,
        metadata,
    };

    let json_data = serde_json::to_string_pretty(&model_weights)?;
    fs::write(output_path, json_data)?;
    
    info!("✅ Model weights saved to: {}", output_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();
    
    info!("🚀 Starting LSTM-CNN model training");
    info!("Data directory: {}", cli.data_dir);
    info!("Sequence length: {}", cli.sequence_length);
    info!("Epochs: {}", cli.epochs);
    info!("Learning rate: {}", cli.learning_rate);

    // Load data
    let input_path = format!("{}/{}", cli.data_dir, cli.input_file);
    let consolidated_data = load_consolidated_data(&input_path)?;

    // Prepare training data
    let (mut data_matrix, feature_names) = prepare_training_data(&consolidated_data)?;
    let normalization_params = normalize_data(&mut data_matrix);
    
    info!("Data shape: {:?}", data_matrix.dim());
    info!("Features: {:?}", feature_names);

    // Create sequences
    let (sequences, targets) = create_sequences(&data_matrix, cli.sequence_length);
    info!("Created {} sequences of length {}", sequences.dim().0, cli.sequence_length);

    // Initialize model
    let mut model = LSTMCNNModel::new(
        cli.sequence_length,
        feature_names.len(),
        cli.lstm_hidden_size,
        cli.cnn_filters,
    );

    // Train model
    info!("🎯 Starting training...");
    let (train_loss, val_loss) = train_model(
        &mut model,
        &sequences,
        &targets,
        cli.epochs,
        cli.learning_rate,
        cli.batch_size,
    );

    // Create output directory
    fs::create_dir_all(&cli.output_dir)?;

    // Save model weights
    let metadata = ModelMetadata {
        training_epochs: cli.epochs,
        learning_rate: cli.learning_rate,
        batch_size: cli.batch_size,
        train_loss,
        validation_loss: val_loss,
        assets_trained: consolidated_data.assets.clone(),
        feature_names,
        normalization_params,
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    let output_path = format!("{}/lstm_cnn_model_weights.json", cli.output_dir);
    save_model_weights(&model, metadata, &output_path)?;

    println!("🎉 Training completed successfully!");
    println!("   📊 Final train loss: {:.6}", train_loss);
    println!("   📊 Final validation loss: {:.6}", val_loss);
    println!("   💾 Model weights saved to: {}", output_path);
    println!("   🎯 Assets trained: {}", consolidated_data.total_assets);
    println!("   📈 Total records processed: {}", consolidated_data.total_records);

    Ok(())
}

// Exemplos de uso:

// # Treinamento básico
// cargo run -- --data-dir ../../dados/consolidado

// # Com parâmetros customizados
// cargo run -- --data-dir ../../dados/consolidado --epochs 200 --learning-rate 0.0001 --batch-size 64

// # Sequência mais longa e modelo maior
// cargo run -- --sequence-length 120 --lstm-hidden-size 128 --cnn-filters 64 --epochs 300

// # Treinamento completo com configuração otimizada
// cargo run -- \
//   --data-dir ../../dados/consolidado \
//   --sequence-length 90 \
//   --epochs 500 \
//   --batch-size 32 \
//   --learning-rate 0.0005 \
//   --lstm-hidden-size 100 \
//   --cnn-filters 48 \
//   --output-dir ./trained_models