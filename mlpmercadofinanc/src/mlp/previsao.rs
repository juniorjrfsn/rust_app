// File : src/mlp/previsao.rs
use crate::conexao::read_file::ler_csv;
use crate::mlp::mlp_cotacao::rna; // Import the rna module
use std::error::Error as StdError;
 
use thiserror::Error;

use crate::mlp::mlp_cotacao::LSTMModel;
use burn::backend::NdArray;

pub fn predict(matrix: Vec<Vec<String>>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model = LSTMModel::<NdArray>::new(64, &NdArray::Device::default());
    Ok(model.predict(matrix)?)
}

#[derive(Debug, Error)]
pub enum PrevCotacaoError {
    #[error("Model not found in database")]
    ModelNotFoundError,
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
    #[error("Failed to parse float: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    #[error("Empty input data")]
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
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    if std == 0.0 {
        return Err(PrevCotacaoError::InvalidDataFormat("Standard deviation is zero".to_string()));
    }
    Ok((mean, std))
}




pub fn parse_row(row: &[String]) -> Result<Vec<f64>, PrevCotacaoError> {
    if row.iter().any(|s| s == "n/d") {
        return Err(PrevCotacaoError::InvalidDataFormat("Missing data in row".to_string()));
    }
    let parse = |s: &str| s.replace(',', ".").replace('%', "").parse::<f64>().map_err(PrevCotacaoError::from);
    Ok(vec![
        parse(&row[1])?,
        parse(&row[3])?,
        parse(&row[4])?,
        parse(&row[5])?,
        parse(&row[6].trim_end_matches(|c| c == 'M' || c == 'B'))? * if row[6].ends_with('B') { 1_000.0 } else { 1.0 },
    ])
}


// Function to make predictions
pub fn prever(
    matrix: Vec<Vec<String>>,
    model: rna::MLP,
    means: &[f64],
    stds: &[f64],
    label_mean: f64,
    label_std: f64,
    _cotac_fonte: &str,
    ativo_financeiro: &str,
) -> Result<(), PrevCotacaoError> {
    let ultima_linha = &matrix[matrix.len() - 1];
    let parsed_row = parse_row(ultima_linha)?;

    // Normalize input for prediction
    let normalized_input: Vec<f64> = parsed_row
        .iter()
        .enumerate()
        .map(|(i, &value)| (value - means[i]) / stds[i])
        .collect();

    // Predict and denormalize
    let predicted_normalized = model.forward(&normalized_input, "tanh")[0];
    let predicted_denormalized = denormalize(predicted_normalized, label_mean, label_std);

    // Log results
    println!(
        "Dados do último registro do CSV: Abertura: {:.2}, Variação: {:.2}%, Mínimo: {:.2}, Máximo: {:.2}, Volume: {:.2}",
        parsed_row[0], parsed_row[1], parsed_row[2], parsed_row[3], parsed_row[4]
    );
    println!(
        "Ativo: {} - Previsão de fechamento para amanhã: {:.2}",
        ativo_financeiro, predicted_denormalized
    );

    Ok(())
}


    // Function to train the MLP model
    pub fn treinar(matrix: Vec<Vec<String>>) -> Result<(rna::MLP, Vec<f64>, Vec<f64>, f64, f64), PrevCotacaoError> {
        if matrix.is_empty() {
            return Err(PrevCotacaoError::EmptyInputData);
        }

        // Parse input data
        let mut inputs = Vec::new();
        let mut labels = Vec::new();
        for row in matrix.iter().skip(1) {
            let parsed_row = parse_row(row)?;
            inputs.push(parsed_row);
            let label = row[2].replace(',', ".").parse::<f64>()?;
            labels.push(label);
        }

        // Normalize inputs
        let mut means = vec![0.0; 5];
        let mut stds = vec![0.0; 5];
        for feature in 0..5 {
            let column: Vec<f64> = inputs.iter().map(|row| row[feature]).collect();
            let (mean, std) = normalize(&column)?;
            means[feature] = mean;
            stds[feature] = std;
            for (input, value) in inputs.iter_mut().zip(column) {
                input[feature] = (value - mean) / std;
            }
        }

        // Normalize labels
        let label_mean = labels.iter().sum::<f64>() / labels.len() as f64;
        let label_std = (labels.iter().map(|x| (x - label_mean).powi(2)).sum::<f64>() / labels.len() as f64).sqrt();

        // Split data into training and testing sets
        let split_idx = (inputs.len() as f64 * 0.8) as usize;
        let (train_inputs, test_inputs) = inputs.split_at(split_idx);
        let (train_labels, test_labels) = labels.split_at(split_idx);

        // Create and train the MLP model
        const ARCHITECTURE: &[usize] = &[5, 64, 32, 16, 8, 1];
        const EPOCHS: usize = 100;
        const LEARNING_RATE: f64 = 0.001;
        const ACTIVATION: &str = "tanh";

        let mut model = rna::MLP::new(ARCHITECTURE);
        model.train(train_inputs, train_labels, EPOCHS, LEARNING_RATE, ACTIVATION);

        // Evaluate the model on test data
        let test_loss = test_inputs.iter().zip(test_labels).fold(0.0, |acc, (input, label)| {
            let prediction = model.forward(input, ACTIVATION)[0];
            acc + (prediction - label).powi(2)
        });
        println!("Test Loss: {:.4}", test_loss / test_inputs.len() as f64);

        Ok((model, means, stds, label_mean, label_std))
    }
// mlp_cotacao.rs