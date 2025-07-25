// file : src/mlp/data_utils.rs

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData}; // Ensure Backend is imported
use thiserror::Error;

/// Custom error types for LSTM operations.
#[derive(Error, Debug)]
pub enum LSTMError {
    /// Represents an invalid data format error.
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    /// Represents an I/O error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Parses a single row of string data into a vector of f32.
///
/// # Arguments
///
/// * `row` - A slice of strings representing a row of financial data.
///
/// # Returns
///
/// A `Result` containing a `Vec<f32>` with parsed numerical data, or an `LSTMError`.
pub fn parse_row(row: &[String]) -> Result<Vec<f32>, LSTMError> {
    // Check if the row has enough columns and no "n/d" values
    if row.len() < 7 || row.iter().any(|s| s == "n/d") {
        return Err(LSTMError::InvalidData(format!("Invalid row: {:?}", row)));
    }

    // Helper closure to parse a string to f32, handling comma as decimal separator
    let parse = |s: &str, field: &str| {
        s.replace(',', ".")
            .parse::<f32>()
            .map_err(|_| LSTMError::InvalidData(format!("Failed to parse {}: {}", field, s)))
    };

    // Parse specific columns: Opening, High, Low, Variation, Volume, Closing
    Ok(vec![
        parse(&row[1], "opening price")?, // Abertura
        parse(&row[5], "high price")?,   // Máximo
        parse(&row[4], "low price")?,    // Mínimo
        parse(&row[3].trim_end_matches('%'), "variation")?, // Variação (remove '%' before parsing)
        parse_volume(&row[6])?,          // Volume (special parsing for 'B', 'M', 'K')
        parse(&row[2], "closing price")?, // Fechamento (used as target)
    ])
}

/// Parses a volume string (e.g., "1.2M", "500K", "3B") into an f32.
///
/// # Arguments
///
/// * `s` - The volume string.
///
/// # Returns
///
/// A `Result` containing the parsed volume as f32, or an `LSTMError`.
pub fn parse_volume(s: &str) -> Result<f32, LSTMError> {
    let s = s.trim();
    // Determine multiplier based on suffix (B for Billion, M for Million, K for Thousand)
    let multiplier = if s.ends_with('B') {
        1e9
    } else if s.ends_with('M') {
        1e6
    } else if s.ends_with('K') {
        1e3
    } else {
        1.0
    };

    // Remove suffix, replace comma with dot, and parse to f32
    s.trim_end_matches(|c| c == 'B' || c == 'M' || c == 'K')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| LSTMError::InvalidData(format!("Invalid volume format: {}", s)))
}

/// Normalizes a row of numerical data using provided means and standard deviations.
///
/// # Arguments
///
/// * `row` - The vector of f32 data to normalize.
/// * `means` - A slice of mean values for each feature.
/// * `stds` - A slice of standard deviation values for each feature.
///
/// # Returns
///
/// A `Vec<f32>` containing the normalized data.
pub fn normalize_row(row: Vec<f32>, means: &[f32], stds: &[f32]) -> Vec<f32> {
    row.iter()
        .enumerate()
        .map(|(i, &x)| (x - means[i]) / stds[i].max(1e-8)) // Add small epsilon to std to prevent division by zero
        .collect()
}

/// Calculates the mean and standard deviation for the first 5 features across all valid rows in the matrix.
///
/// # Arguments
///
/// * `matrix` - A slice of string vectors representing the raw financial data.
///
/// # Returns
///
/// A `Result` containing a tuple of `(Vec<f32>, Vec<f32>)` for means and stds, or an `LSTMError`.
pub fn calculate_stats(matrix: &[Vec<String>]) -> Result<(Vec<f32>, Vec<f32>), LSTMError> {
    let mut data = Vec::new();
    // Parse each row and collect valid numerical data
    for row in matrix {
        match parse_row(row) {
            Ok(parsed) => data.push(parsed),
            Err(_) => continue, // Skip rows that cannot be parsed
        }
    }

    if data.is_empty() {
        return Err(LSTMError::InvalidData("No valid data found for stats calculation.".into()));
    }

    // Calculate means for the first 5 features (Opening, High, Low, Variation, Volume)
    let means = (0..5)
        .map(|i| data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32)
        .collect::<Vec<_>>();

    // Calculate standard deviations for the first 5 features
    let stds = (0..5)
        .map(|i| {
            let variance = data
                .iter()
                .map(|row| (row[i] - means[i]).powi(2))
                .sum::<f32>()
                / data.len() as f32;
            variance.sqrt()
        })
        .collect::<Vec<_>>();
    Ok((means, stds))
}

/// Preprocesses the raw financial data matrix into sequences and targets for LSTM.
///
/// # Arguments
///
/// * `matrix` - A slice of string vectors representing the raw financial data.
/// * `seq_length` - The length of each input sequence.
/// * `device` - The Burn device to create tensors on.
///
/// # Returns
///
/// A `Result` containing a tuple:
/// (Tensor<B, 3> for input sequences, Vec<String> for dates, f32 for target mean, f32 for target std)
/// or an `LSTMError`.
pub fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
    seq_length: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Vec<String>, f32, f32), LSTMError> {
    if matrix.len() < seq_length + 1 {
        return Err(LSTMError::InvalidData(format!(
            "Not enough data for sequence length {}. Required: {}, Got: {}",
            seq_length,
            seq_length + 1,
            matrix.len()
        )));
    }

    let (means, stds) = calculate_stats(matrix)?;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    let mut dates = Vec::new();

    // Create sequences and collect targets and dates
    for i in 0..matrix.len().saturating_sub(seq_length) {
        let seq: Result<Vec<Vec<f32>>, LSTMError> = matrix[i..i + seq_length]
            .iter()
            .map(|row| Ok(normalize_row(parse_row(row)?, &means, &stds)))
            .collect();
        let seq = seq?;
        let target = parse_row(&matrix[i + seq_length])?[2]; // Closing price is at index 2 after parsing
        sequences.push(seq);
        targets.push(target);
        dates.push(matrix[i + seq_length][0].clone()); // Date is at index 0 of the raw row
    }

    if sequences.is_empty() {
        return Err(LSTMError::InvalidData("No sequences generated after preprocessing.".into()));
    }

    // Calculate mean and std for target values for denormalization
    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets
        .iter()
        .map(|&x| (x - target_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32)
        .sqrt();

    // Flatten sequences and create a 3D tensor
    let x = Tensor::from_floats(
        TensorData::new(
            sequences
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<f32>>(),
            Shape::new([dates.len(), seq_length, 5]), // [batch_size, sequence_length, num_features]
        ),
        device,
    );

    Ok((x, dates, target_mean, target_std))
}

/// Preprocesses the raw financial data matrix into sequences and normalized targets for training.
///
/// This version also returns normalized targets for training purposes.
///
/// # Arguments
///
/// * `matrix` - A slice of string vectors representing the raw financial data.
/// * `seq_length` - The length of each input sequence.
/// * `device` - The Burn device to create tensors on.
///
/// # Returns
///
/// A `Result` containing a tuple:
/// (Tensor<B, 3> for input sequences, Tensor<B, 2> for normalized targets,
/// Vec<String> for dates, f32 for target mean, f32 for target std)
/// or an `LSTMError`.
pub fn preprocess_for_training<B: Backend>(
    matrix: &[Vec<String>],
    seq_length: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2>, Vec<String>, f32, f32), LSTMError> {
    if matrix.len() < seq_length + 1 {
        return Err(LSTMError::InvalidData(format!(
            "Not enough data for sequence length {}. Required: {}, Got: {}",
            seq_length,
            seq_length + 1,
            matrix.len()
        )));
    }

    let (means, stds) = calculate_stats(matrix)?;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    let mut dates = Vec::new();

    // Create sequences and collect targets and dates
    for i in 0..matrix.len().saturating_sub(seq_length) {
        let seq: Result<Vec<Vec<f32>>, LSTMError> = matrix[i..i + seq_length]
            .iter()
            .map(|row| Ok(normalize_row(parse_row(row)?, &means, &stds)))
            .collect();
        let seq = seq?;
        let target = parse_row(&matrix[i + seq_length])?[2]; // Closing price is at index 2 after parsing
        sequences.push(seq);
        targets.push(target);
        dates.push(matrix[i + seq_length][0].clone()); // Date is at index 0 of the raw row
    }

    if sequences.is_empty() {
        return Err(LSTMError::InvalidData("No sequences generated after preprocessing.".into()));
    }

    // Calculate mean and std for target values for denormalization
    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets
        .iter()
        .map(|&x| (x - target_mean).powi(2))
        .sum::<f32>()
        / targets.len() as f32)
        .sqrt();

    // Normalize targets for training
    let normalized_targets: Vec<f32> = targets
        .into_iter()
        .map(|x| (x - target_mean) / target_std.max(1e-8))
        .collect();

    // Flatten sequences and create a 3D tensor
    let x = Tensor::from_floats(
        TensorData::new(
            sequences
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<f32>>(),
            Shape::new([dates.len(), seq_length, 5]), // [batch_size, sequence_length, num_features]
        ),
        device,
    );

    // Create a 2D tensor for normalized targets
    let y = Tensor::from_floats(
        TensorData::new(normalized_targets, Shape::new([dates.len(), 1])), // [batch_size, 1]
        device,
    );

    Ok((x, y, dates, target_mean, target_std))
}
