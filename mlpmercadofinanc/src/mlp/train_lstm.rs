use burn::{
    module::Module,
    nn::{
        loss::MSELoss,
        rnn::{LSTM, LSTMConfig},
        Linear, LinearConfig,
    },
    tensor::{
        backend::Backend,
        Tensor,
        Data,
        Shape,
    },
    train::{
        LearnerBuilder,
        metric::LossMetric,
    },
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    #[error("Tensor error: {0}")]
    Tensor(#[from] burn::tensor::Error),
}

pub struct LSTMModel<B: Backend> {
    lstm: LSTM<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config_lstm = LSTMConfig::new(5, 64); // 5 features
        let lstm = LSTM::new(device, config_lstm);
        
        let config_linear = LinearConfig::new(64, 1);
        let linear = Linear::new(device, config_linear);
        
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let outputs = self.lstm.forward(inputs);
        let last_output = outputs.select(1, -1);
        self.linear.forward(last_output)
    }
}

pub fn treinar<B: Backend>(matrix: Vec<Vec<String>>, device: &B::Device) -> Result<(), LSTMError> {
    let (x_train, y_train) = preprocess::<B>(matrix)?;

    let model = LSTMModel::new(device);
    let loss_fn = MSELoss::new();
    let mut optim = burn::optim::Adam::new(0.001);

    for epoch in 1..=100 {
        let outputs = model.forward(x_train.clone());
        let loss = loss_fn.forward(outputs, y_train.clone());
        
        optim.backward_step(&loss);
        
        println!("Epoch: {epoch}, Loss: {:.4}", loss.to_data().value[0]);
    }

    Ok(())
}

fn preprocess<B: Backend>(matrix: Vec<Vec<String>>) -> Result<(Tensor<B, 3>, Tensor<B, 2>), LSTMError> {
    let seq_length = 30;
    let (means, stds) = calculate_stats(&matrix)?;

    let mut sequences = Vec::new();
    let mut targets = Vec::new();

    for i in 0..matrix.len() - seq_length {
        let seq = matrix[i..i+seq_length].iter()
            .map(|row| normalize_row(parse_row(row)?, &means, &stds))
            .collect::<Result<Vec<_>, _>>()?;
        
        let target = parse_row(&matrix[i+seq_length])?[2]; // Último (fechamento)
        
        sequences.push(seq);
        targets.push(target);
    }

    // Cria tensores com Burn
    let x = Tensor::from_data(
        Data::new(sequences.into_iter().flatten().collect(), Shape::new([sequences.len(), seq_length, 5]))
    ).to_device(device);

    let y = Tensor::from_data(
        Data::new(targets, Shape::new([targets.len(), 1]))
    ).to_device(device);

    Ok((x, y))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    let parse = |s: &str| -> Result<f32, LSTMError> {
        s.replace(',', ".")
            .parse::<f32>()
            .map_err(|_| LSTMError::InvalidData("Failed to parse number".into()))
    };

    Ok(vec![
        parse(&row[2])?,    // Abertura
        parse(&row[3])?,    // Máxima
        parse(&row[4])?,    // Mínima
        parse(&row[6].trim_end_matches('%'))?, // Variação
        parse_volume(&row[5])? // Volume
    ])
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let multiplier = if s.ends_with('B') { 1e9 }
    else if s.ends_with('M') { 1e6 }
    else { 1.0 };
    
    s.trim_end_matches(|c| c == 'B' || c == 'M')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| LSTMError::InvalidData("Invalid volume format".into()))
}

fn normalize_row(row: Vec<f32>, means: &[f32], stds: &[f32]) -> Vec<f32> {
    row.iter().enumerate()
        .map(|(i, &x)| (x - means[i]) / stds[i])
        .collect()
}

fn calculate_stats(matrix: &[Vec<String>]) -> Result<(Vec<f32>, Vec<f32>), LSTMError> {
    let mut data = Vec::new();
    for row in matrix {
        data.push(parse_row(row)?);
    }
    
    let means = (0..5).map(|i| {
        data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32
    }).collect::<Vec<_>>();

    let stds = (0..5).map(|i| {
        let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }).collect::<Vec<_>>();

    Ok((means, stds))
}