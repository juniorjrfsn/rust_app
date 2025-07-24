use burn::tensor::{TensorData, Shape, Tensor};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use log::warn;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Invalid data format at {location}: {message}")]
    InvalidData { location: String, message: String },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("Model loading error: {0}")]
    ModelLoadError(String),
}

#[derive(Serialize, Deserialize)]
pub struct NormalizationParams {
    pub target_mean: f32,
    pub target_std: f32,
}

pub fn parse_row(row: &[String]) -> Result<Vec<f32>, AppError> {
    if row.len() < 7 || row.iter().any(|s| s == "n/d") {
        return Err(AppError::InvalidData {
            location: "row".to_string(),
            message: format!("Invalid row: {:?}", row),
        });
    }
    let parse = |s: &str, field: &str| s.replace(',', ".").parse::<f32>()
        .map_err(|_| AppError::InvalidData {
            location: field.to_string(),
            message: format!("Failed to parse {}: {}", field, s),
        });
    Ok(vec![
        parse(&row[1], "opening price")?,
        parse(&row[5], "high price")?,
        parse(&row[4], "low price")?,
        parse(&row[3].trim_end_matches('%'), "variation")?,
        parse_volume(&row[6])?,
        parse(&row[2], "closing price")?,
    ])
}

pub fn parse_volume(s: &str) -> Result<f32, AppError> {
    let s = s.trim();
    let multiplier = if s.ends_with('B') { 1e9 } else if s.ends_with('M') { 1e6 } else if s.ends_with('K') { 1e3 } else { 1.0 };
    s.trim_end_matches(|c| c == 'B' || c == 'M' || c == 'K')
        .replace(',', ".")
        .parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| AppError::InvalidData {
            location: "volume".to_string(),
            message: format!("Invalid volume format: {}", s),
        })
}

pub fn normalize_row(row: Vec<f32>, means: &[f32], stds: &[f32]) -> Vec<f32> {
    row.iter().enumerate().map(|(i, &x)| (x - means[i]) / stds[i].max(1e-8)).collect()
}

pub fn calculate_stats(matrix: &[Vec<String>]) -> Result<(Vec<f32>, Vec<f32>), AppError> {
    let mut data = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        match parse_row(row) {
            Ok(parsed) => data.push(parsed),
            Err(e) => warn!("Skipping row {}: {}", i + 1, e),
        }
    }
    if data.is_empty() {
        return Err(AppError::InvalidData {
            location: "matrix".to_string(),
            message: "No valid data found".to_string(),
        });
    }
    let means = (0..5).map(|i| data.iter().map(|row| row[i]).sum::<f32>() / data.len() as f32).collect::<Vec<_>>();
    let stds = (0..5).map(|i| {
        let variance = data.iter().map(|row| (row[i] - means[i]).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }).collect::<Vec<_>>();
    Ok((means, stds))
}

pub fn preprocess<B: Backend>(
    matrix: &[Vec<String>],
    seq_length: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2>, Vec<String>, f32, f32), AppError> {
    let (means, stds) = calculate_stats(matrix)?;
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    let mut dates = Vec::new();

    for i in 0..matrix.len().saturating_sub(seq_length) {
        let seq: Result<Vec<Vec<f32>>, AppError> = matrix[i..i + seq_length]
            .iter()
            .map(|row| Ok(normalize_row(parse_row(row)?, &means, &stds)))
            .collect();
        let seq = seq?;
        let target = parse_row(&matrix[i + seq_length])?[2];
        sequences.push(seq);
        targets.push(target);
        dates.push(matrix[i + seq_length][0].clone());
    }

    if sequences.is_empty() {
        return Err(AppError::InvalidData {
            location: "preprocess".to_string(),
            message: "No sequences generated".to_string(),
        });
    }

    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets.iter().map(|&x| (x - target_mean).powi(2)).sum::<f32>() / targets.len() as f32).sqrt();
    let normalized_targets: Vec<f32> = targets.iter().map(|&x| (x - target_mean) / target_std).collect();

    let x = Tensor::from_floats(
        TensorData::new(
            sequences.into_iter().flatten().flatten().collect::<Vec<f32>>(),
            Shape::new([targets.len(), seq_length, 5]),
        ),
        device,
    );
    let y = Tensor::from_floats(
        TensorData::new(normalized_targets, Shape::new([targets.len(), 1])),
        device,
    );

    Ok((x, y, dates, target_mean, target_std))
}

pub fn save_normalization_params(mean: f32, std: f32, path: &str) -> Result<(), AppError> {
    let params = NormalizationParams { target_mean: mean, target_std: std };
    let file = std::fs::File::create(path).map_err(|e| AppError::IoError(e))?;
    serde_json::to_writer(file, &params).map_err(|e| AppError::InvalidData {
        location: "normalization params".to_string(),
        message: e.to_string(),
    })?;
    Ok(())
}

pub fn load_normalization_params(path: &str) -> Result<(f32, f32), AppError> {
    let file = std::fs::File::open(path).map_err(|e| AppError::IoError(e))?;
    let params: NormalizationParams = serde_json::from_reader(file).map_err(|e| AppError::InvalidData {
        location: "normalization params".to_string(),
        message: e.to_string(),
    })?;
    Ok((params.target_mean, params.target_std))
}