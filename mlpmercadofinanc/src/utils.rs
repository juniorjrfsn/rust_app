// File : src/utils.rs 



use serde::Deserialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("Invalid data format: {0}")]
    InvalidData(String),
    #[error("Invalid date format: {0}")]
    InvalidDate(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Recorder error: {0}")]
    RecorderError(#[from] burn::record::RecorderError),
}

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    #[serde(rename = "Data")]
    pub data: String,
    #[serde(rename = "Abertura")]
    pub abertura: String,
    #[serde(rename = "Fechamento")]
    pub fechamento: String,
    #[serde(rename = "Máxima")]
    pub maximo: String,
    #[serde(rename = "Mínima")]
    pub minimo: String,
    #[serde(rename = "Vol.")]
    pub volume: String,
    #[serde(rename = "Var%")]
    pub variacao: String,
}

pub fn parse_volume(s: &str) -> Result<f32, AppError> {
    let s = s.trim();
    if s.is_empty() || s == "n/d" {
        return Err(AppError::InvalidData(format!("Invalid volume string: {}", s)));
    }

    let (number_str, multiplier) = if s.ends_with('B') {
        (s.trim_end_matches('B'), 1e9)
    } else if s.ends_with('M') {
        (s.trim_end_matches('M'), 1e6)
    } else if s.ends_with('K') {
        (s.trim_end_matches('K'), 1e3)
    } else {
        (s, 1.0)
    };

    let cleaned = number_str.replace(',', ".");
    cleaned.parse::<f32>()
        .map(|v| v * multiplier)
        .map_err(|_| AppError::InvalidData(format!("Invalid volume format: {}", s)))
}

pub fn parse_row(record: &Record) -> Result<Vec<f32>, AppError> {
    let parse = |s: &str, field: &str| -> Result<f32, AppError> {
        let cleaned = s.replace(',', ".").trim().to_string();
        if cleaned.is_empty() || cleaned == "n/d" {
            return Err(AppError::InvalidData(format!("Invalid value for {}: {}", field, s)));
        }
        cleaned.parse::<f32>()
            .map_err(|_| AppError::InvalidData(format!("Failed to parse {}: {}", field, s)))
    };

    Ok(vec![
        parse(&record.abertura, "opening")?,
        parse(&record.maximo, "high")?,
        parse(&record.minimo, "low")?,
        parse_volume(&record.volume)?,
        parse(&record.variacao.trim_end_matches('%'), "variation")?,
        parse(&record.fechamento, "closing")?,
    ])
}