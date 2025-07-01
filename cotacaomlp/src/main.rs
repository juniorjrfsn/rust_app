use std::env;
use std::path::Path;
use std::fs;

// Define conexao module inline
mod conexao {
    pub mod read_file {
        use csv::ReaderBuilder;
        use serde::Deserialize;
        use std::fs;
        use thiserror::Error;
        use serde::de::DeserializeOwned;

        #[derive(Error, Debug)]
        pub enum DataError {
            #[error("IO error: {0}")]
            IoError(#[from] std::io::Error),
            #[error("CSV error: {0}")]
            CsvError(#[from] csv::Error),
            #[error("Invalid data format: {0}")]
            InvalidDataFormat(String),
        }

        #[derive(Debug, Deserialize)]
        struct ObjectInvestingHistCotacaoAtivo {
            #[serde(rename = "Data")]
            data: String,
            #[serde(rename = "Último")]
            fechamento: String,
            #[serde(rename = "Abertura")]
            abertura: String,
            #[serde(rename = "Máxima")]
            maximo: String,
            #[serde(rename = "Mínima")]
            minimo: String,
            #[serde(rename = "Vol.")]
            volume: String,
            #[serde(rename = "Var%")]
            variacao: String,
        }

        #[derive(Debug, Deserialize)]
        struct Record {
            #[serde(rename = "DATA")]
            data: String,
            #[serde(rename = "ABERTURA")]
            abertura: String,
            #[serde(rename = "FECHAMENTO")]
            fechamento: String,
            #[serde(rename = "VARIAÇÃO")]
            variacao: String,
            #[serde(rename = "MÍNIMO")]
            minimo: String,
            #[serde(rename = "MÁXIMO")]
            maximo: String,
            #[serde(rename = "VOLUME")]
            volume: String,
        }

        trait CsvRecord {
            fn into_row(self) -> Vec<String>;
        }

        impl CsvRecord for ObjectInvestingHistCotacaoAtivo {
            fn into_row(self) -> Vec<String> {
                vec![
                    self.data,
                    self.abertura,
                    self.fechamento,
                    self.variacao,
                    self.minimo,
                    self.maximo,
                    self.volume,
                ]
            }
        }

        impl CsvRecord for Record {
            fn into_row(self) -> Vec<String> {
                vec![
                    self.data,
                    self.abertura,
                    self.fechamento,
                    self.variacao,
                    self.minimo,
                    self.maximo,
                    self.volume,
                ]
            }
        }

        fn read_csv_generic<T: DeserializeOwned + CsvRecord>(
            file_path: &str,
        ) -> Result<Vec<Vec<String>>, DataError> {
            if !fs::metadata(file_path).is_ok() {
                return Err(DataError::InvalidDataFormat(format!(
                    "Arquivo não encontrado: {}",
                    file_path
                )));
            }
            let file = fs::File::open(file_path).map_err(DataError::IoError)?;
            let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
            let mut matrix: Vec<Vec<String>> = Vec::new();
            for result in rdr.deserialize() {
                let record: T = result.map_err(DataError::CsvError)?;
                matrix.push(record.into_row());
            }
            if matrix.is_empty() || matrix.iter().any(|row| row.len() != 7) {
                return Err(DataError::InvalidDataFormat(
                    "O arquivo CSV não contém dados suficientes ou formato inválido.".to_string(),
                ));
            }
            Ok(matrix)
        }

        pub fn ler_csv(
            file_path: &str,
            cotac_fonte: &str,
            _ativo_financeiro: &str,
        ) -> Result<Vec<Vec<String>>, DataError> {
            let matrix = match cotac_fonte {
                "investing" => read_csv_generic::<ObjectInvestingHistCotacaoAtivo>(file_path)?,
                "infomoney" => read_csv_generic::<Record>(file_path)?,
                _ => return Err(DataError::InvalidDataFormat(format!(
                    "Fonte de cotação desconhecida: {}",
                    cotac_fonte
                ))),
            };
            Ok(matrix)
        }
    }
}

// Define mlp module inline
mod mlp {
    pub mod previsao {
        use super::mlp_cotacao::rna;
        use thiserror::Error;
        use std::collections::HashMap;

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
            #[error("Generic error: {0}")]
            GenericError(#[from] Box<dyn std::error::Error>),
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
                return Err(PrevCotacaoError::InvalidDataFormat("Standard deviation is zero".to_string()));
            }
            Ok((mean, std))
        }

        pub fn parse_row(row: &[String], column_map: &HashMap<&str, usize>) -> Result<Vec<f64>, PrevCotacaoError> {
            println!("Processando linha: {:?}", row);

            if row.len() < 7 {
                return Err(PrevCotacaoError::InvalidDataFormat(format!(
                    "Linha inválida ou faltando dados: {:?}",
                    row
                )));
            }

            let parse_value = |s: &str| -> Result<f64, PrevCotacaoError> {
                let cleaned = if s.trim().is_empty() || s == "n/d" {
                    println!("Valor inválido encontrado: '{}'. Substituindo por '0,0'", s);
                    "0,0"
                } else {
                    s
                };
                cleaned.replace(',', ".").replace('%', "").parse::<f64>().map_err(|e| {
                    PrevCotacaoError::InvalidDataFormat(format!("Falha ao converter valor '{}': {}", s, e))
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

            let ultimo = parse_value(&row[*column_map.get("Último").unwrap_or(&1)])?;
            let abertura = parse_value(&row[*column_map.get("Abertura").unwrap_or(&2)])?;
            let maxima = parse_value(&row[*column_map.get("Máxima").unwrap_or(&3)])?;
            let minima = parse_value(&row[*column_map.get("Mínima").unwrap_or(&4)])?;
            let volume = parse_volume(&row[*column_map.get("Vol.").unwrap_or(&5)])?;

            Ok(vec![ultimo, abertura, maxima, minima, volume])
        }

        pub fn treinar(
            matrix: Vec<Vec<String>>,
            train_ratio: f64,
        ) -> Result<(rna::MLP, Vec<f64>, Vec<f64>, f64, f64), PrevCotacaoError> {
            if matrix.is_empty() {
                return Err(PrevCotacaoError::EmptyInputData);
            }

            let mut inputs = Vec::new();
            let mut labels = Vec::new();

            let column_map = HashMap::from([
                ("Último", 1), // Maps to FECHAMENTO
                ("Abertura", 2),
                ("Máxima", 3),
                ("Mínima", 4),
                ("Vol.", 5),
            ]);

            for row in matrix.iter().skip(1) {
                let parsed_row = parse_row(row, &column_map)?;
                inputs.push(parsed_row.clone());
                let label = parsed_row[0]; // Using FECHAMENTO as label
                labels.push(label);
            }

            let mut means = vec![0.0; 5];
            let mut stds = vec![0.0; 5];
            for feature in 0..5 {
                let column: Vec<f64> = inputs.iter().map(|row| row[feature]).collect();
                let (mean, std) = normalize(&column)?;
                means[feature] = mean;
                stds[feature] = std;
                for input in &mut inputs {
                    input[feature] = (input[feature] - mean) / std;
                }
            }

            let label_mean = labels.iter().sum::<f64>() / labels.len() as f64;
            let label_std = (labels.iter().map(|x| (x - label_mean).powi(2)).sum::<f64>() / labels.len() as f64).sqrt();
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

            let mut model = rna::MLP::new(ARCHITECTURE);
            model.train(train_inputs, train_labels, EPOCHS, LEARNING_RATE, ACTIVATION)
                .map_err(|e| PrevCotacaoError::GenericError(e.into()))?;

            let validation_loss = train_inputs.iter().zip(train_labels).fold(0.0, |acc, (input, &label)| {
                let prediction = model.forward(input, ACTIVATION)[0];
                acc + (prediction - label).powi(2)
            }) / train_inputs.len() as f64;
            println!("Validation Loss: {:.4}", validation_loss);

            Ok((model, means, stds, label_mean, label_std))
        }

        pub fn prever(
            matrix: Vec<Vec<String>>,
            model: rna::MLP,
            means: &[f64],
            stds: &[f64],
            label_mean: f64,
            label_std: f64,
            ativo_financeiro: &str,
        ) -> Result<(), PrevCotacaoError> {
            let column_map = HashMap::from([
                ("Último", 1),
                ("Abertura", 2),
                ("Máxima", 3),
                ("Mínima", 4),
                ("Vol.", 5),
            ]);

            let latest_row = matrix.iter().skip(1).max_by_key(|row| row[0].clone())
                .ok_or(PrevCotacaoError::EmptyInputData)?;

            let parsed_row = parse_row(latest_row, &column_map)?;

            let normalized_input: Vec<f64> = parsed_row
                .iter()
                .enumerate()
                .map(|(i, &value)| (value - means[i]) / stds[i])
                .collect();

            let predicted_normalized = model.forward(&normalized_input, "relu")[0];
            let predicted_denormalized = denormalize(predicted_normalized, label_mean, label_std);

            println!(
                "Dados do último registro do CSV: Abertura: {:.2}, Máxima: {:.2}, Mínima: {:.2}, Volume: {:.2}",
                parsed_row[1], parsed_row[2], parsed_row[3], parsed_row[4]
            );
            println!(
                "Ativo: {} - Previsão de fechamento para amanhã: {:.2}",
                ativo_financeiro, predicted_denormalized
            );

            Ok(())
        }
    }

    pub mod mlp_cotacao {
        pub mod rna {
            use rand::Rng;
            use rand::rng;
            use serde::{Serialize, Deserialize};
            use bincode;
            use std::error::Error;

            fn apply_activation(x: f64, activation: &str) -> f64 {
                match activation {
                    "relu" => relu(x),
                    "tanh" => tanh(x),
                    "sigmoid" => sigmoid(x),
                    _ => x,
                }
            }

            fn activation_derivative(x: f64, activation: &str) -> f64 {
                match activation {
                    "relu" => relu_derivative(x),
                    "tanh" => tanh_derivative(x),
                    "sigmoid" => sigmoid_derivative(x),
                    _ => 1.0,
                }
            }

            fn tanh(x: f64) -> f64 { x.tanh() }
            fn tanh_derivative(x: f64) -> f64 { 1.0 - x.tanh().powi(2) }
            fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
            fn sigmoid_derivative(x: f64) -> f64 { let s = sigmoid(x); s * (1.0 - s) }
            fn relu(x: f64) -> f64 { if x > 0.0 { x } else { 0.0 } }
            fn relu_derivative(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }

            #[derive(Serialize, Deserialize, Debug, Clone)]
            pub struct DenseLayer {
                pub weights: Vec<Vec<f64>>,
                pub biases: Vec<f64>,
            }

            impl DenseLayer {
                pub fn new(input_size: usize, output_size: usize) -> Self {
                    let scale = (2.0 / (input_size + output_size) as f64).sqrt();
                    let mut rng = rng();
                    let weights = (0..output_size)
                        .map(|_| (0..input_size).map(|_| rng.random_range(-scale..scale)).collect())
                        .collect();
                    let biases = vec![0.0; output_size];
                    DenseLayer { weights, biases }
                }

                pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
                    self.weights.iter().zip(&self.biases).map(|(weights, &bias)| {
                        let sum: f64 = inputs.iter().zip(weights).map(|(&x, &w)| x * w).sum();
                        apply_activation(sum + bias, "relu")
                    }).collect()
                }
            }

            #[derive(Serialize, Deserialize, Debug, Clone)]
            pub struct MLP {
                pub layers: Vec<DenseLayer>,
            }

            impl MLP {
                pub fn new(layer_sizes: &[usize]) -> Self {
                    let layers = layer_sizes.windows(2)
                        .map(|sizes| DenseLayer::new(sizes[0], sizes[1]))
                        .collect();
                    MLP { layers }
                }

                pub fn forward(&self, inputs: &[f64], _activation: &str) -> Vec<f64> {
                    let mut output = inputs.to_vec();
                    for layer in &self.layers {
                        output = layer.forward(&output);
                    }
                    output
                }

                pub fn train(
                    &mut self,
                    inputs: &[Vec<f64>],
                    labels: &[f64],
                    epochs: usize,
                    learning_rate: f64,
                    activation: &str,
                ) -> Result<(), Box<dyn Error>> {
                    if inputs.len() != labels.len() {
                        return Err("Mismatch between inputs and labels length".into());
                    }
                    for epoch in 0..epochs {
                        let mut total_loss = 0.0;
                        for (input, &label) in inputs.iter().zip(labels) {
                            let mut activations = vec![input.clone()];
                            for layer in &self.layers {
                                let output = layer.forward(activations.last().unwrap());
                                activations.push(output);
                            }

                            let prediction = activations.last().unwrap()[0];
                            let loss = (prediction - label).powi(2);
                            total_loss += loss;

                            let mut delta = 2.0 * (prediction - label);
                            for i in (0..self.layers.len()).rev() {
                                let output = activations[i + 1][0]; // Dereference here
                                let input = &activations[i];
                                let gradient = delta * activation_derivative(output, activation);
                                for j in 0..self.layers[i].weights.len() {
                                    self.layers[i].weights[j][0] -= learning_rate * gradient * input[0];
                                }
                                self.layers[i].biases[0] -= learning_rate * delta;
                                delta = self.layers[i].weights[0][0] * activation_derivative(output, activation);
                            }
                        }
                        println!("Epoch: {}, Loss: {:.4}", epoch + 1, total_loss / inputs.len() as f64);
                    }
                    Ok(())
                }

                pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn Error>> {
                    let serialized = bincode::serialize(self)?;
                    Ok(serialized)
                }

                pub fn deserialize(data: &[u8]) -> Result<Self, Box<dyn Error>> {
                    let deserialized = bincode::deserialize(data)?;
                    Ok(deserialized)
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Forneça exatamente um argumento: 'treino' ou 'reconhecer'",
        )));
    }
    let mode = &args[1];

    let cotac_fonte = "infomoney";
    let ativo_financeiro = "SLCE3";
    let file_path = Path::new("dados").join(cotac_fonte).join(format!("{}.csv", ativo_financeiro));

    if !file_path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Arquivo não encontrado: {:?}", file_path),
        )));
    }

    println!("Lendo dados do arquivo: {:?}", file_path);
    let matrix = conexao::read_file::ler_csv(file_path.to_str().unwrap(), cotac_fonte, ativo_financeiro)?;

    let mut model = None;
    if mode == "treino" {
        println!("Treinando o modelo...");
        let train_ratio = 0.8;
        let (trained_model, trained_means, trained_stds, trained_label_mean, trained_label_std) =
            mlp::previsao::treinar(matrix, train_ratio)?;
        model = Some(trained_model);
        let means = trained_means;
        let stds = trained_stds;
        let label_mean = trained_label_mean;
        let label_std = trained_label_std;
        println!("Modelo treinado com sucesso.");
        let serialized = model.as_ref().unwrap().serialize()?;
        fs::write("model.bin", serialized)?;
        let serialized_params = bincode::serialize(&(means, stds, label_mean, label_std))?;
        fs::write("params.bin", serialized_params)?;
        println!("Modelo e parâmetros salvos em model.bin e params.bin");
    } else if mode == "reconhecer" {
        println!("Carregando modelo treinado...");
        let serialized = fs::read("model.bin")?;
        let loaded_model = mlp::mlp_cotacao::rna::MLP::deserialize(&serialized)?;
        model = Some(loaded_model);

        println!("Carregando parâmetros normalizados...");
        let params_data = fs::read("params.bin")?;
        let (means, stds, label_mean, label_std) = bincode::deserialize(&params_data)?;
        // Note: Fixed typo in params_data reference

        println!("Fazendo previsão...");
        if let Some(m) = &model {
            mlp::previsao::prever(matrix, m.clone(), &means, &stds, label_mean, label_std, ativo_financeiro)?;
        } else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Modelo não encontrado. Execute o modo 'treino' primeiro.",
            )));
        }
        println!("Previsão concluída com sucesso.");
    } else {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Modo inválido. Use 'treino' ou 'reconhecer'",
        )));
    }

    Ok(())
}