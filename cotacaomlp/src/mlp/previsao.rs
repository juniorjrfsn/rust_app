use crate::conexao::read_file::{ler_csv, DataError};
use crate::mlp::mlp_cotacao::rna::{self, MLP};
use std::collections::HashMap;
use thiserror::Error;
use std::num::ParseFloatError;

#[derive(Error, Debug)]
pub enum PrevCotacaoError {
    #[error("Modelo não encontrado no banco de dados")]
    ModelNotFoundError,
    #[error("Formato de dados inválido: {0}")]
    InvalidDataFormat(String),
    #[error("Erro ao converter float: {0}")]
    ParseFloatError(#[from] ParseFloatError),
    #[error("Dados de entrada vazios")]
    EmptyInputData,
}

pub fn denormalize(value: f64, mean: f64, std: f64) -> f64 {
    value * std + mean
}

pub fn normalize(data: &[f64]) -> Result<(f64, f64), PrevCotacaoError> {
    if data.is_empty() {
        return Err(PrevCotacaoError::EmptyInputData);
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = variance.sqrt();
    if std == 0.0 {
        return Err(PrevCotacaoError::InvalidDataFormat("Desvio padrão é zero".to_string()));
    }
    Ok((mean, std))
}

pub fn parse_row(row: &[String], column_map: &HashMap<&str, usize>) -> Result<Vec<f64>, PrevCotacaoError> {
    if row.len() < 7 {
        return Err(PrevCotacaoError::InvalidDataFormat("Linha inválida".to_string()));
    }

    let parse_value = |s: &str| {
        let cleaned = if s.trim().is_empty() || s == "n/d" { "0,0" } else { s };
        cleaned.replace(',', ".").replace('%', "").parse::<f64>().map_err(|_| {
            PrevCotacaoError::InvalidDataFormat(format!("Falha ao converter valor '{}'", s))
        })
    };

    let parse_volume = |s: &str| -> Result<f64, PrevCotacaoError> {
        let value = parse_value(s.trim_end_matches(|c| c == 'M' || c == 'B' || c == 'K'))?;
        let multiplier = match s.chars().last() {
            Some('B') => 1_000_000_000.0,
            Some('M') => 1_000_000.0,
            Some('K') => 1_000.0,
            _ => 1.0,
        };
        Ok(value * multiplier)
    };

    Ok(vec![
        parse_value(&row[*column_map.get("Último").unwrap_or(&1)])?,
        parse_value(&row[*column_map.get("Abertura").unwrap_or(&2)])?,
        parse_value(&row[*column_map.get("Máxima").unwrap_or(&3)])?,
        parse_value(&row[*column_map.get("Mínima").unwrap_or(&4)])?,
        parse_volume(&row[*column_map.get("Vol.").unwrap_or(&5)])?,
    ])
}

pub fn treinar(
    matrix: Vec<Vec<String>>,
    train_ratio: f64,
) -> Result<(MLP, Vec<f64>, Vec<f64>, f64, f64), PrevCotacaoError> {
    if matrix.is_empty() {
        return Err(PrevCotacaoError::EmptyInputData);
    }

    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    let column_map = HashMap::from([
        ("Último", 1),
        ("Abertura", 2),
        ("Máxima", 3),
        ("Mínima", 4),
        ("Vol.", 5),
    ]);

    for row in matrix.iter().skip(1) {
        let parsed_row = parse_row(row, &column_map)?;
        inputs.push(parsed_row.clone());
        labels.push(parsed_row[0]);
    }

    let mut means = vec![0.0; 5];
    let mut stds = vec![0.0; 5];

    for feature in 0..5 {
        let col: Vec<f64> = inputs.iter().map(|r| r[feature]).collect();
        let (m, s) = normalize(&col)?;
        means[feature] = m;
        stds[feature] = s;
        for input in &mut inputs {
            input[feature] = (input[feature] - m) / s;
        }
    }

    let label_mean = labels.iter().sum::<f64>() / labels.len() as f64;
    let label_std = labels.iter().map(|x| (x - label_mean).powi(2)).sum::<f64>().sqrt();

    for label in &mut labels {
        *label = (*label - label_mean) / label_std;
    }

    let split_idx = (inputs.len() as f64 * train_ratio) as usize;
    let train_inputs = &inputs[..split_idx];
    let train_labels = &labels[..split_idx];

    const ARCHITECTURE: &[usize] = &[5, 64, 32, 1];
    const EPOCHS: usize = 300;
    const LEARNING_RATE: f64 = 0.0001;
    const ACTIVATION: &str = "relu";

    let mut model = MLP::new(ARCHITECTURE);
    model.train(train_inputs, train_labels, EPOCHS, LEARNING_RATE, ACTIVATION)?;

    Ok((model, means, stds, label_mean, label_std))
}

pub fn prever(
    matrix: Vec<Vec<String>>,
    model: MLP,
    means: &[f64],
    stds: &[f64],
    label_mean: f64,
    label_std: f64,
    ativo: &str,
) -> Result<(), PrevCotacaoError> {
    let column_map = HashMap::from([
        ("Último", 1),
        ("Abertura", 2),
        ("Máxima", 3),
        ("Mínima", 4),
        ("Vol.", 5),
    ]);

    let latest_row = matrix.first().ok_or(PrevCotacaoError::EmptyInputData)?;
    let parsed = parse_row(latest_row, &column_map)?;

    let normalized: Vec<f64> = parsed.iter().enumerate()
        .map(|(i, &v)| (v - means[i]) / stds[i])
        .collect();

    let pred = model.forward(&normalized, "relu")[0];
    let final_pred = pred * label_std + label_mean;

    println!("Previsão para {}: {:.2}", ativo, final_pred);
    Ok(())
}