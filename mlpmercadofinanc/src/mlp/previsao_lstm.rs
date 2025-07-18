// src/mlp/previsao_lstm.rs


use burn::{
    module::Module,
    nn::{
        loss::MseLoss,
        Linear, LinearConfig,
        LSTM, LSTMConfig,
    },
    optim::AdamConfig,
    tensor::{backend::Backend, Data, Shape, Tensor},
    train::{LearnerBuilder, metric::LossMetric},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    #[error("Tensor error: {0}")]
    Tensor(#[from] burn::tensor::TensorError),
}

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: LSTM<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config_lstm = LSTMConfig::new(5, 64); // 5 features, 64 hidden units
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(64, 1); // 64 input features, 1 output
        let linear = config_linear.init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        let last_output = outputs.slice([0..outputs.dims()[0], outputs.dims()[1] - 1..outputs.dims()[1]]);
        self.linear.forward(last_output)
    }
}

pub fn treinar<B: Backend>(matrix: Vec<Vec<String>>, device: &B::Device) -> Result<(), LSTMError> {
    let (x_train, y_train) = preprocess::<B>(matrix, device)?;
    let model = LSTMModel::new(device);
    let loss_fn = MseLoss::new();
    let config = AdamConfig::new().with_learning_rate(0.001);
    let optim = config.init();
    let learner = LearnerBuilder::new(".")
        .devices(vec![device.clone()])
        .num_epochs(100)
        .build(model, optim, loss_fn);
    for epoch in 1..=100 {
        let state = learner.step(x_train.clone(), y_train.clone());
        println!("Epoch: {epoch}, Loss: {:.4}", state.loss.to_data().value[0]);
    }
    Ok(())
}

fn preprocess<B: Backend>(
    matrix: Vec<Vec<String>>,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2>), LSTMError> {
    let seq_length = 30;
    let (means, stds) = calculate_stats(&matrix)?;
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
    let x = Tensor::from_data(
        Data::new(
            sequences.into_iter().flatten().collect(),
            Shape::new([sequences.len() as i64, seq_length as i64, 5]),
        ),
        device,
    );
    let y = Tensor::from_data(
        Data::new(
            targets,
            Shape::new([targets.len() as i64, 1]),
        ),
        device,
    );
    Ok((x, y))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    let parse = |s: &str| -> Result<f32, LSTMError> {
        s.replace(',', ".")
            .parse::<f32>()
            .map_err(|_| LSTMError::InvalidData("Failed to parse number".into()))
    };
    Ok(vec![
        parse(&row[1])?, // Open (Abertura)
        parse(&row[5])?, // High (Máxima)
        parse(&row[4])?, // Low (Mínima)
        parse(&row[3].trim_end_matches('%'))?, // Change % (Var%)
        parse_volume(&row[6])?, // Volume
    ])
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let multiplier = if s.ends_with('B') {
        1e9
    } else if s.ends_with('M') {
        1e6
    } else {
        1.0
    };
    s.trim_end_matches(|c| c == 'B' || c == 'M')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| LSTMError::InvalidData("Invalid volume format".into()))
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
    let means = (0..5)
        .map(|i| {
            data.iter().map(|row| row[i]).sum::<f32>() / data.len() as attendant
        })
        .collect::<Vec<_>>();
    let stds = (0..5)
        .map(|i| {
            let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>()
                / data.len() as f32;
            variance.sqrt()
        })
        .collect::<Vec<_>>();
    Ok((means, stds))
}

//