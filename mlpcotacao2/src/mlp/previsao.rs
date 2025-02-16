// previsao.rs 
use crate::mlp::mlp_cotacao::rna; // Import the rna module
use thiserror::Error; // Import the Error trait
use std::collections::HashMap;
 

#[derive(Debug, Error)]
pub enum PrevCotacaoError {
    /*
    #[error("Model not found in database")]
    ModelNotFoundError,
    */
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

pub fn parse_row(row: &[String], column_map: &HashMap<&str, usize>) -> Result<Vec<f64>, PrevCotacaoError> {
    print!("\rProcessando linha: {:?}", row); // Log da linha atual

    if row.len() < 7 {
        return Err(PrevCotacaoError::InvalidDataFormat(format!(
            "Linha inválida ou faltando dados: {:?}",
            row
        )));
    }

    let parse_value = |s: &str| {
        let cleaned = if s.trim().is_empty() || s == "n/d" {
            println!("Valor inválido encontrado: '{}'. Substituindo por '0,0'", s);
            "0,0" // Substitui valores inválidos por "0,0"
        } else {
            s
        };
        cleaned.replace(',', ".").replace('%', "").parse::<f64>().map_err(|_| {
            PrevCotacaoError::InvalidDataFormat(format!("Falha ao converter valor: {}", s))
        })
    };

    let parse_volume = |s: &str| -> Result<f64, PrevCotacaoError> {
        let value = parse_value(s.trim_end_matches(|c| c == 'M' || c == 'B' || c == 'K'))?;
        let multiplier = if s.ends_with('B') {
            1_000_000_000.0
        } else if s.ends_with('M') {
            1_000_000.0
        } else if s.ends_with('K') {
            1_000.0
        } else {
            1.0
        };
        Ok(value * multiplier)
    };

    // Use the column map to dynamically access columns
    let ultimo = parse_value(&row[*column_map.get("Último").unwrap_or(&1)])?;
    let abertura = parse_value(&row[*column_map.get("Abertura").unwrap_or(&3)])?;
    let maxima = parse_value(&row[*column_map.get("Máxima").unwrap_or(&4)])?;
    let minima = parse_value(&row[*column_map.get("Mínima").unwrap_or(&5)])?;
    let volume = parse_volume(&row[*column_map.get("Vol.").unwrap_or(&6)])?;

    Ok(vec![ultimo, abertura, maxima, minima, volume])
}

// Function to train the MLP model
pub fn treinar(
    matrix: Vec<Vec<String>>,
    train_ratio: f64,
) -> Result<(rna::MLP, Vec<f64>, Vec<f64>, f64, f64), PrevCotacaoError> {
    if matrix.is_empty() {
        return Err(PrevCotacaoError::EmptyInputData);
    }

    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    // Criar o mapa de colunas
    let mut column_map = HashMap::new();
    column_map.insert("Último", 1);
    column_map.insert("Abertura", 2); // Ajuste os índices conforme necessário
    column_map.insert("Máxima", 4);
    column_map.insert("Mínima", 5);
    column_map.insert("Vol.", 6);

    for row in matrix.iter().skip(1) {
        let parsed_row = parse_row(row, &column_map)?;
        inputs.push(parsed_row);
        let label = row[2].replace(',', ".").parse::<f64>()?;
        labels.push(label);
    }
    println!("");
    // Normalizar os dados de entrada
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
    println!("");
    // Normalizar os rótulos
    let label_mean = labels.iter().sum::<f64>() / labels.len() as f64;
    let label_std = (labels.iter().map(|x| (x - label_mean).powi(2)).sum::<f64>() / labels.len() as f64).sqrt();

    // Dividir os dados em conjuntos de treino e teste
    let split_idx = (inputs.len() as f64 * train_ratio) as usize;
    let (train_inputs, test_inputs) = inputs.split_at(split_idx);
    let (train_labels, test_labels) = labels.split_at(split_idx);

    // Criar e treinar o modelo MLP
    //const ARCHITECTURE: &[usize] = &[5, 128, 64, 32, 16, 1]; // Aumenta a complexidade
    //const EPOCHS: usize = 200;
    //const LEARNING_RATE: f64 = 0.001;
    //const ACTIVATION: &str = "tanh";
    

    // Criar e treinar o modelo MLP
    const ARCHITECTURE: &[usize] = &[5, 64, 32, 1]; // Simplifica a arquitetura
    const EPOCHS: usize = 300; // Aumenta o número de épocas
    const LEARNING_RATE: f64 = 0.0001; // Reduz a taxa de aprendizado
    const ACTIVATION: &str = "relu"; // Usa ReLU como função de ativação

    let mut model = rna::MLP::new(ARCHITECTURE);
    model.train(train_inputs, train_labels, EPOCHS, LEARNING_RATE, ACTIVATION);

    // Avaliar o modelo no conjunto de validação
    let validation_loss = test_inputs.iter().zip(test_labels).fold(0.0, |acc, (input, label)| {
        let prediction = model.forward(input, ACTIVATION)[0];
        acc + (prediction - label).powi(2)
    });
    println!("Validation Loss: {:.4}", validation_loss / test_inputs.len() as f64);

    Ok((model, means, stds, label_mean, label_std))
}

// Function to make predictions
pub fn prever(
    matrix: Vec<Vec<String>>,
    model: rna::MLP,
    means: &[f64],
    stds: &[f64],
    label_mean: f64,
    label_std: f64,
    ativo_financeiro: &str,
) -> Result<(), PrevCotacaoError> {
    // Criar o mapa de colunas
    let mut column_map = HashMap::new();
    column_map.insert("Último", 1);
    column_map.insert("Abertura", 3);
    column_map.insert("Máxima", 4);
    column_map.insert("Mínima", 5);
    column_map.insert("Vol.", 6);

    let cotacao_mais_recente = &matrix[0]; // ultima_linha = &matrix[matrix.len() - 1];
    println!("");
    let parsed_row = parse_row(cotacao_mais_recente, &column_map)?;
    println!("");
    let normalized_input: Vec<f64> = parsed_row
        .iter()
        .enumerate()
        .map(|(i, &value)| (value - means[i]) / stds[i])
        .collect();

    let predicted_normalized = model.forward(&normalized_input, "tanh")[0];
    let predicted_denormalized = denormalize(predicted_normalized, label_mean, label_std);

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
// mlp_cotacao.rs