// File : src/mlp/train_lstm.rs

use burn::{
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Lstm, LstmConfig},
    optim::{AdamConfig, Optimizer},
    tensor::{backend::Backend, TensorData, Shape, Tensor},
    train::{
        LearnerBuilder, 
        metric::{LossMetric, LearningRateMetric},
    },
    data::dataset::Dataset,
};
use thiserror::Error;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use plotly::{Plot, Scatter, Layout};

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
}

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(hidden_size: usize, device: &B::Device) -> Self {
        let config_lstm = LstmConfig::new(6, hidden_size, true);
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(hidden_size, 1);
        let linear = config_linear.init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        let last_output = outputs.slice([0..outputs.dims()[0], outputs.dims()[1] - 1..outputs.dims()[1]]);
        let last_output_2d = last_output.reshape([last_output.dims()[0], last_output.dims()[2]]);
        self.linear.forward(last_output_2d)
    }
}

#[derive(Debug, Clone)]
struct FinancialDataset {
    x: Vec<Vec<Vec<f32>>>,
    y: Vec<f32>,
}

impl<B: Backend> Dataset<(Tensor<B, 3>, Tensor<B, 2>)> for FinancialDataset {
    fn get(&self, index: usize) -> Option<(Tensor<B, 3>, Tensor<B, 2>)> {
        self.x.get(index).map(|seq| {
            let x = Tensor::from_floats(
                TensorData::new(seq.clone().into_iter().flatten().flatten().collect::<Vec<f32>>(), Shape::new([1, seq.len(), seq[0].len()])),
                &B::Device::default(),
            );
            let y = Tensor::from_floats(
                TensorData::new(vec![self.y[index]], Shape::new([1, 1])),
                &B::Device::default(),
            );
            (x, y)
        })
    }

    fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn treinar<B: Backend>(matrix: Vec<Vec<String>>, _device: &B::Device) -> Result<(), LSTMError> {
    if matrix.len() < 31 {
        return Err(LSTMError::InvalidData("Not enough data for sequence length".into()));
    }

    let (x_train, y_train, x_valid, y_valid, target_mean, target_std) = preprocess::<B>(&matrix)?;
    
    let model = LSTMModel::new(64, &B::Device::default());
    let loss_fn = MseLoss::new();
    let config = AdamConfig::new();
    let optim = config.init();

    let mut best_loss = f32::INFINITY;
    let mut patience = 10;
    let mut early_stop_counter = 0;

    let learner = LearnerBuilder::new(".")
        .devices(vec![B::Device::default()])
        .num_epochs(100)
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train(LearningRateMetric::new())
        .build(model, optim, loss_fn);

    let train_dataset = FinancialDataset {
        x: x_train.clone(),
        y: y_train.clone().into_iter().map(|y| y[0]).collect(),
    };
    let valid_dataset = FinancialDataset {
        x: x_valid,
        y: y_valid.into_iter().map(|y| y[0]).collect(),
    };

    let mut losses_train = Vec::new();
    let mut losses_valid = Vec::new();

    for epoch in 1..=100 {
        let state = learner.step(&train_dataset, &valid_dataset);
        let train_loss = state.train_metrics.get::<LossMetric>().unwrap().value();
        let valid_loss = state.valid_metrics.get::<LossMetric>().unwrap().value();
        losses_train.push(train_loss);
        losses_valid.push(valid_loss);
        println!("Epoch: {}, Train Loss: {:.4}, Valid Loss: {:.4}", epoch, train_loss, valid_loss);

        if valid_loss < best_loss {
            best_loss = valid_loss;
            early_stop_counter = 0;
        } else {
            early_stop_counter += 1;
            if early_stop_counter >= patience {
                println!("Early stopping triggered at epoch {}", epoch);
                break;
            }
        }
    }

    save_loss_plot(&losses_train, &losses_valid);

    Ok(())
}

fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, f32, f32), LSTMError> {
    let seq_length = 30;
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
    let target_std = (targets.iter().map(|&x| (x - target_mean).powi(2)).sum::<f32>() / targets.len() as f32).sqrt();
    let targets: Vec<f32> = targets.into_iter().map(|x| (x - target_mean) / target_std.max(1e-8)).collect();

    let split_idx = (sequences.len() as f32 * 0.8) as usize;
    let mut indices: Vec<usize> = (0..sequences.len()).collect();
    let mut rng = StdRng::seed_from_u64(42);
    indices.shuffle(&mut rng);

    let train_indices = &indices[..split_idx];
    let valid_indices = &indices[split_idx..];

    let x_train: Vec<_> = train_indices.iter().map(|&i| sequences[i].clone()).collect();
    let y_train: Vec<_> = train_indices.iter().map(|&i| vec![targets[i]]).collect();
    let x_valid: Vec<_> = valid_indices.iter().map(|&i| sequences[i].clone()).collect();
    let y_valid: Vec<_> = valid_indices.iter().map(|&i| vec![targets[i]]).collect();

    Ok((x_train, y_train, x_valid, y_valid, target_mean, target_std))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    let parse = |s: &str| -> Result<f32, LSTMError> {
        s.replace(',', ".")
            .parse::<f32>()
            .map_err(|_| LSTMError::InvalidData(format!("Failed to parse number: {}", s)))
    };
    Ok(vec![
        parse(&row[2])?, // Fechamento
        parse(&row[1])?, // Abertura
        parse(&row[5])?, // Máximo
        parse(&row[4])?, // Mínimo
        parse(&row[3].trim_end_matches('%'))?, // Variação
        parse_volume(&row[6])?, // Volume
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
    let means = (0..6)
        .map(|i| {
            data.iter().map(`row| row[i]).sum::<f32>() / data.len() as f32
        })
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

fn save_loss_plot(losses_train: &[f32], losses_valid: &[f32]) {
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
        .title("LSTM Training and Validation Loss".into())
        .x_axis(plotly::layout::Axis::new().title("Epoch".into()))
        .y_axis(plotly::layout::Axis::new().title("Loss".into()));
    plot.set_layout(layout);

    plot.write_html("loss_plot.html");
}