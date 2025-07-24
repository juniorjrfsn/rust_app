// file: src/mlp/previsao_lstm.rs


use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Lstm, LstmConfig},
    tensor::{backend::Backend, TensorData, Shape, Tensor},
};
use thiserror::Error;
use plotly::{Plot, Scatter, Layout};

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(hidden_size: usize, device: &B::Device) -> Self {
        let config_lstm = LstmConfig::new(5, hidden_size, true);
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(hidden_size * 2, 1);
        let linear = config_linear.init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        let outputs = outputs.clone(); // Fix move error
        let last_output = outputs.slice([0..outputs.dims()[0], outputs.dims()[1] - 1..outputs.dims()[1]]);
        let last_output = last_output.clone(); // Fix move error
        let last_output_2d = last_output.reshape([last_output.dims()[0], last_output.dims()[2]]);
        self.linear.forward(last_output_2d)
    }
}

pub fn predict<B: Backend>(
    matrix: &Vec<Vec<String>>,
    device: &B::Device,
    model_path: &str,
) -> Result<Vec<(String, f32)>, LSTMError> {
    let (x, dates, _, _, target_mean, target_std) = preprocess::<B>(matrix, device)?;
    let mut model = LSTMModel::<B>::new(64, device);
    // TODO: Load model weights
    // model.load(model_path)?;
    let output = model.forward(x);
    let output_data = output.to_data().to_vec::<f32>();
    let predictions: Vec<f32> = output_data.into_iter().map(|x| x * target_std + target_mean).collect();
    let results: Vec<(String, f32)> = dates.into_iter().zip(predictions).collect();
    plot_predictions(&results, matrix)?;
    Ok(results)
}

fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Vec<String>, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, f32, f32), LSTMError> {
    let seq_length = 30;
    let (means, stds) = calculate_stats(matrix)?;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    let mut dates = Vec::new();

    for i in 0..matrix.len().saturating_sub(seq_length) {
        let seq: Result<Vec<Vec<f32>>, LSTMError> = matrix[i..i + seq_length]
            .iter()
            .map(|row| Ok(normalize_row(parse_row(row)?, &means, &stds)))
            .collect();
        let seq = seq?;
        let target = parse_row(&matrix[i + seq_length])?[2]; // Closing price
        sequences.push(seq);
        targets.push(target);
        dates.push(matrix[i + seq_length][0].clone());
    }

    if sequences.is_empty() {
        return Err(LSTMError::InvalidData("No sequences generated".into()));
    }

    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets.iter().map(|&x| (x - target_mean).powi(2)).sum::<f32>() / targets.len() as f32).sqrt();

    let x = Tensor::from_floats(
        TensorData::new(sequences.clone().into_iter().flatten().flatten().collect::<Vec<f32>>(), Shape::new([sequences.len(), seq_length, 5])),
        device,
    );

    Ok((x, dates, vec![], vec![], target_mean, target_std))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    if row.len() < 7 || row.iter().any(|s| s == "n/d") {
        return Err(LSTMError::InvalidData(format!("Invalid row: {:?}", row)));
    }
    let parse = |s: &str, field: &str| s.replace(',', ".").parse::<f32>()
        .map_err(|_| LSTMError::InvalidData(format!("Failed to parse {}: {}", field, s)));
    Ok(vec![
        parse(&row[1], "opening price")?, // Abertura
        parse(&row[5], "high price")?,   // Máximo
        parse(&row[4], "low price")?,    // Mínimo
        parse(&row[3].trim_end_matches('%'), "variation")?, // Variação
        parse_volume(&row[6])?,          // Volume
        parse(&row[2], "closing price")?, // Fechamento (used as target)
    ])
}

fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let s = s.trim();
    let multiplier = if s.ends_with('B') { 1e9 } else if s.ends_with('M') { 1e6 } else if s.ends_with('K') { 1e3 } else { 1.0 };
    s.trim_end_matches(|c| c == 'B' || c == 'M' || c == 'K')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| LSTMError::InvalidData(format!("Invalid volume format: {}", s)))
}

fn normalize_row(row: Vec<f32>, means: &[f32], stds: &[f32]) -> Vec<f32> {
    row.iter().enumerate().map(|(i, &x)| (x - means[i]) / stds[i].max(1e-8)).collect()
}

fn calculate_stats(matrix: &[Vec<String>]) -> Result<(Vec<f32>, Vec<f32>), LSTMError> {
    let mut data = Vec::new();
    for row in matrix {
        match parse_row(row) {
            Ok(parsed) => data.push(parsed),
            Err(_) => continue,
        }
    }
    if data.is_empty() {
        return Err(LSTMError::InvalidData("No valid data found".into()));
    }
    let means = (0..5).map(|i| data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32).collect::<Vec<_>>();
    let stds = (0..5).map(|i| {
        let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }).collect::<Vec<_>>();
    Ok((means, stds))
}

fn plot_predictions(predictions: &[(String, f32)], matrix: &[Vec<String>]) -> Result<(), LSTMError> {
    let mut plot = Plot::new();
    let mut actual_prices = Vec::new();
    let mut dates = Vec::new();

    for row in matrix.iter().skip(30) {
        let price = parse_row(row)?[2];
        actual_prices.push(price);
        dates.push(row[0].clone());
    }

    let pred_prices: Vec<f32> = predictions.iter().map(|(_, p)| *p).collect();
    let trace_actual = Scatter::new(dates.clone(), actual_prices)
        .name("Actual Price")
        .mode(plotly::common::Mode::Lines);
    let trace_pred = Scatter::new(dates, pred_prices)
        .name("Predicted Price")
        .mode(plotly::common::Mode::Lines);

    plot.add_trace(trace_actual);
    plot.add_trace(trace_pred);
    let layout = Layout::new()
        .title("WEGE3 Actual vs Predicted Prices")
        .x_axis(plotly::layout::Axis::new().title("Date"))
        .y_axis(plotly::layout::Axis::new().title("Price (BRL)"));
    plot.set_layout(layout);
    plot.write_html("predictions_plot.html");
    println!("Predictions plot saved to predictions_plot.html");
    Ok(())
}