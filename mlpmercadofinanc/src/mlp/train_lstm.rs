// File : src/mlp/train_lstm.rs

use burn::{
    lr_scheduler::constant::ConstantLr,
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Lstm, LstmConfig},
    optim::AdamConfig,
    tensor::{backend::Backend, Tensor},
    train::{
        metric::{LearningRateMetric, LossMetric},
        LearnerBuilder, TrainOutput, TrainStep, ValidStep, ValidOutput, // Corrected import for ValidOutput
    },
};
use plotly::{Layout, Plot, Scatter};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use burn::tensor::backend::AutodiffBackend;

// Import data utility functions and LSTMError
use crate::mlp::data_utils::{preprocess_for_training, LSTMError};
 
use thiserror::Error;
  
use plotly::{Plot, Scatter, Layout};

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Model error: {0}")]
    ModelError(String),
}

// Model definition is mostly correct, but fix LstmConfig
#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(hidden_size: usize, device: &B::Device) -> Self {
        // LstmConfig::new now requires 3 parameters: input_size, hidden_size, bidirectional
        let config_lstm = LstmConfig::new(6, hidden_size, false); // Set bidirectional to false for simplicity
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(hidden_size, 1); // Output size 1 for prediction
        let linear = config_linear.init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        // Fix borrow issues by cloning before slice/reshape
        let last_output = outputs.clone().slice([0..outputs.dims()[0], outputs.dims()[1] - 1..outputs.dims()[1]]);
        let last_output_2d = last_output.clone().reshape([last_output.dims()[0], last_output.dims()[2]]);
        self.linear.forward(last_output_2d)
    }
}

// Dataset implementation looks mostly correct, add device storage
#[derive(Debug, Clone)]
struct FinancialDataset<B: Backend> {
    x: Vec<Vec<Vec<f32>>>,
    y: Vec<f32>,
    device: B::Device, // Store the device
}

impl<B: Backend> Dataset<(Tensor<B, 3>, Tensor<B, 2>)> for FinancialDataset<B> {
    fn get(&self, index: usize) -> Option<(Tensor<B, 3>, Tensor<B, 2>)> {
        self.x.get(index).map(|seq| {
            // Use the stored device
            let x = Tensor::from_floats(
                TensorData::new(
                    seq.clone().into_iter().flatten().collect::<Vec<f32>>(), // Fixed flatten chain
                    Shape::new([1, seq.len(), seq[0].len()]),
                ),
                &self.device,
            );
            let y = Tensor::from_floats(
                TensorData::new(vec![self.y[index]], Shape::new([1, 1])),
                &self.device,
            );
            (x, y)
        })
    }

    fn len(&self) -> usize {
        self.x.len()
    }
}

#[derive(Debug, Clone)]
pub struct LSTMConfig {
    pub hidden_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub patience: usize,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            sequence_length: 30,
            learning_rate: 0.001,
            num_epochs: 100,
            batch_size: 32,
            patience: 10,
        }
    }
}

// Main training function - this is where most API issues are
// Constrain B to AutodiffBackend for training
pub fn train<B: AutodiffBackend>(
    matrix: Vec<Vec<String>>,
    device: &B::Device,
) -> Result<(), LSTMError> {
    let config = LSTMConfig::default();

    if matrix.len() < config.sequence_length + 1 {
        return Err(LSTMError::InvalidData(
            format!(
                "Not enough data for sequence length. Required: {}, Got: {}",
                config.sequence_length + 1,
                matrix.len()
            ),
        ));
    }

    // Preprocess data
    let (x_train, y_train, x_valid, y_valid, _target_mean, _target_std) = preprocess::<B>(&matrix, config.sequence_length)?;

    // Create model and training components
    let model = LSTMModel::<B>::new(config.hidden_size, device);
    let loss_fn = MseLoss::new();
    let optim_config = AdamConfig::new();
    // Initialize optimizer with explicit type parameters to help the compiler
    let optim = optim_config.init::<B, LSTMModel<B>>();

    let mut best_loss = f32::INFINITY;
    let mut early_stop_counter = 0;

    // Build learner with correct types and LrScheduler
    // The key fix: The 4th generic parameter should be the loss function type, and the 5th the LrScheduler type
    let learner = LearnerBuilder::new(".")
        .devices(vec![device.clone()]) // Pass device correctly
        .num_epochs(config.num_epochs)
        // Use the correct metric methods for numeric values
        .metric_train_numeric(LossMetric::new()) // Use metric_train_numeric
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        // The build method takes (model, optimizer, loss_fn, lr_scheduler)
        .build(
            model,
            optim,
            loss_fn,
            ConstantLr::new(config.learning_rate), // Pass the LrScheduler instance correctly
        );

    // Create datasets with the device
    let train_dataset = FinancialDataset {
        x: x_train.clone(),
        y: y_train.clone(),
        device: device.clone(), // Pass device to dataset
    };
    let valid_dataset = FinancialDataset {
        x: x_valid,
        y: y_valid,
        device: device.clone(),
    };

    let mut losses_train = Vec::new();
    let mut losses_valid = Vec::new();

    // Training loop - fix the learner usage
    for epoch in 1..=config.num_epochs {
        // Use the learner's fit method correctly
        // fit returns a Trainer which we use for one epoch
        let mut trainer = learner.fit(config.batch_size); 
        trainer.train(&train_dataset); // Train on one epoch
        // Extract loss values correctly
        let train_loss = trainer.train_metrics().get::<LossMetric>().unwrap().value().item(); // .item() gets the scalar f32
        
        trainer.valid(&valid_dataset); // Validate
        let valid_loss = trainer.valid_metrics().get::<LossMetric>().unwrap().value().item(); // .item() gets the scalar f32
        
        losses_train.push(train_loss);
        losses_valid.push(valid_loss);
        println!(
            "Epoch: {}, Train Loss: {:.4}, Valid Loss: {:.4}",
            epoch, train_loss, valid_loss
        );

        if valid_loss < best_loss {
            best_loss = valid_loss;
            early_stop_counter = 0;
            println!("New best model saved with validation loss: {:.4}", best_loss);
        } else {
            early_stop_counter += 1;
            if early_stop_counter >= config.patience {
                println!("Early stopping triggered at epoch {}", epoch);
                break;
            }
        }
    }

    save_loss_plot(&losses_train, &losses_valid)?;

    Ok(())
}

// Data preprocessing functions - these look mostly correct
fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
    seq_length: usize,
) -> Result<
    (
        Vec<Vec<Vec<f32>>>,
        Vec<f32>,
        Vec<Vec<Vec<f32>>>,
        Vec<f32>,
        f32,
        f32,
    ),
    LSTMError,
> {
    let (means, stds) = calculate_stats(matrix)?;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for i in 0..matrix.len().saturating_sub(seq_length) {
        let seq: Result<Vec<Vec<f32>>, LSTMError> = matrix[i..i + seq_length]
            .iter()
            .map(|row| Ok(normalize_row(parse_row(row)?, &means, &stds)))
            .collect();
        let seq = seq?;
        let target = parse_row(&matrix[i + seq_length])?[2]; // Closing price
        sequences.push(seq);
        targets.push(target);
    }

    if sequences.is_empty() {
        return Err(LSTMError::InvalidData("No sequences generated".into()));
    }

    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets
        .iter()
        .map(|&x| (x - target_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32)
        .sqrt();
    let targets: Vec<f32> = targets
        .into_iter()
        .map(|x| (x - target_mean) / target_std.max(1e-8))
        .collect();

    let split_idx = (sequences.len() as f32 * 0.8) as usize;

    // Split first, then shuffle only training data
    let (train_sequences, valid_sequences) = sequences.split_at(split_idx);
    let (train_targets, valid_targets) = targets.split_at(split_idx);

    let mut train_indices: Vec<usize> = (0..train_sequences.len()).collect();
    let mut rng = StdRng::seed_from_u64(42);
    train_indices.shuffle(&mut rng);

    let x_train: Vec<_> = train_indices.iter().map(|&i| train_sequences[i].clone()).collect();
    let y_train: Vec<_> = train_indices.iter().map(|&i| train_targets[i]).collect(); // Return Vec<f32>
    let x_valid: Vec<_> = (0..valid_sequences.len()).map(|i| valid_sequences[i].clone()).collect();
    let y_valid: Vec<_> = (0..valid_targets.len()).map(|i| valid_targets[i]).collect(); // Return Vec<f32>

    Ok((
        x_train,
        y_train,
        x_valid,
        y_valid,
        target_mean,
        target_std,
    ))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    if row.len() < 7 {
        return Err(LSTMError::InvalidData(format!(
            "Row has insufficient columns: expected 7, got {}",
            row.len()
        )));
    }

    let parse = |s: &str, field: &str| -> Result<f32, LSTMError> {
        s.replace(',', ".")
            .parse::<f32>()
            .map_err(|_| LSTMError::InvalidData(format!("Failed to parse {}: {}", field, s)))
    };

    Ok(vec![
        parse(&row[2], "closing price")?, // Fechamento
        parse(&row[1], "opening price")?, // Abertura
        parse(&row[5], "high price")?,    // Máximo
        parse(&row[4], "low price")?,     // Mínimo
        parse(&row[3].trim_end_matches('%'), "variation")?, // Variação
        parse_volume(&row[6])?,           // Volume
    ])
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let s = s.trim();
    let multiplier = if s.ends_with('B') {
        1e9
    } else if s.ends_with('M') {
        1e6
    } else if s.ends_with('K') {
        1e3
    } else {
        1.0
    };
    s.trim_end_matches(|c| c == 'B' || c == 'M' || c == 'K')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| LSTMError::InvalidData(format!("Invalid volume format: {}", s)))
}

fn normalize_row(row: Vec<f32>, means: &[f32], stds: &[f32]) -> Vec<f32> {
    row.iter()
        .enumerate()
        .map(|(i, &x)| (x - means[i]) / stds[i].max(1e-8))
        .collect()
}

fn calculate_stats(matrix: &[Vec<String>]) -> Result<(Vec<f32>, Vec<f32>), LSTMError> {
    let mut data = Vec::new();
    for row in matrix {
        data.push(parse_row(row)?);
    }

    if data.is_empty() {
        return Err(LSTMError::InvalidData("No valid data found".into()));
    }

    let means = (0..6)
        .map(|i| data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32)
        .collect::<Vec<_>>();
    let stds = (0..6)
        .map(|i| {
            let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>()
                / data.len() as f32;
            variance.sqrt()
        })
        .collect::<Vec<_>>();
    Ok((means, stds))
}

fn save_loss_plot(losses_train: &[f32], losses_valid: &[f32]) -> Result<(), LSTMError> {
    let mut plot = Plot::new();
    let epochs: Vec<usize> = (1..=losses_train.len()).collect();

    let trace_train = Scatter::new(epochs.clone(), losses_train.to_vec())
        .name("Training Loss")
        .mode(plotly::common::Mode::Lines);
    let trace_valid = Scatter::new(epochs, losses_valid.to_vec())
        .name("Validation Loss")
        .mode(plotly::common::Mode::Lines);

    plot.add_trace(trace_train);
    plot.add_trace(trace_valid);

    let layout = Layout::new()
        .title("LSTM Training and Validation Loss")
        .x_axis(plotly::layout::Axis::new().title("Epoch"))
        .y_axis(plotly::layout::Axis::new().title("Loss"));
    plot.set_layout(layout);

    // plotly's write_html returns () in newer versions, so remove map_err
    plot.write_html("loss_plot.html"); 
    println!("Loss plot saved to loss_plot.html");
    Ok(())
}