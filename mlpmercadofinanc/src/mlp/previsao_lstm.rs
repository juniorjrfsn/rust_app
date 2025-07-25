// file : src/mlp/previsao_lstm.rs


use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Lstm, LstmConfig},
    tensor::{backend::Backend, TensorData, Shape, Tensor},
};
use thiserror::Error;

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
        // Fix LstmConfig constructor - it needs 3 parameters
        let config_lstm = LstmConfig::new(6, hidden_size, false); // Set bidirectional to false
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(hidden_size, 1);
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

    pub fn predict(&self, matrix: Vec<Vec<String>>, device: &B::Device) -> Result<Vec<f32>, LSTMError> {
        let (x, _, _, _, target_mean, target_std) = preprocess::<B>(&matrix, device)?;
        let output = self.forward(x);
        
        // Fix tensor data access - use to_data() correctly
        let output_data = output.to_data(); // This returns TensorData
        
        // Process the raw bytes to extract f32 values
        // The data is stored as bytes, we need to convert them back to f32
        let float_count = output_data.shape.num_elements();
        let mut output_vec = Vec::with_capacity(float_count);
        
        // Convert bytes to f32 values
        let bytes = &output_data.bytes;
        for i in 0..float_count {
            let start = i * std::mem::size_of::<f32>();
            let end = start + std::mem::size_of::<f32>();
            if end <= bytes.len() {
                let bytes_slice = &bytes[start..end];
                let float_val = f32::from_le_bytes([
                    bytes_slice[0], bytes_slice[1], bytes_slice[2], bytes_slice[3]
                ]);
                // Apply inverse normalization
                output_vec.push(float_val * target_std + target_mean);
            }
        }
        
        Ok(output_vec)
    }
}

fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, f32, f32), LSTMError> {
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
    let _targets_normalized: Vec<f32> = targets.into_iter().map(|x| (x - target_mean) / target_std.max(1e-8)).collect();

    // Fix borrow issue by cloning sequences before consuming
    let x = Tensor::from_floats(
        TensorData::new(
            sequences.clone().into_iter().flatten().flatten().collect::<Vec<f32>>(), 
            Shape::new([sequences.len(), seq_length, 6]) // Use 6 features
        ),
        device,
    );

    Ok((x, vec![], vec![], vec![], target_mean, target_std))
}

fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    if row.len() < 7 {
        return Err(LSTMError::InvalidData(format!("Row has insufficient columns: expected 7, got {}", row.len())));
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
        // Handle possible errors in parsing rows
        match parse_row(row) {
            Ok(parsed_row) => data.push(parsed_row),
            Err(_) => continue, // Ignore rows with parsing errors
        }
    }
    
    if data.is_empty() {
        return Err(LSTMError::InvalidData("No valid data found".into()));
    }
    
    let means = (0..6) // Use 6 features
        .map(|i| {
            data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32
        })
        .collect::<Vec<_>>();
    let stds = (0..6) // Use 6 features
        .map(|i| {
            let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>()
                / data.len() as f32;
            variance.sqrt()
        })
        .collect::<Vec<_>>();
    Ok((means, stds))
}