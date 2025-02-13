use std::error::Error;
mod conexao;
use crate::conexao::read_file::ler_csv;
use crate::conexao::mlp_cotacao::rna;

// Função para normalizar os dados
fn normalize(data: &mut [f64]) -> (f64, f64) {
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let std: f64 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    for x in data {
        *x = (*x - mean) / std;
    }
    (mean, std)
}

// Função para desnormalizar os dados
fn denormalize(value: f64, mean: f64, std: f64) -> f64 {
    value * std + mean
}

fn main() -> Result<(), Box<dyn Error>> {
    // Carrega os dados do CSV
    let matrix = ler_csv("dados/WEGE3.csv")?;

    // Converte os dados para formato numérico
    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    for row in matrix {
        if row[1] == "n/d" || row[2] == "n/d" {
            continue;
        }
        let abertura: f64 = row[1].replace(',', ".").parse()?;
        let fechamento: f64 = row[2].replace(',', ".").parse()?;
        let variacao: f64 = row[3].replace(',', ".").parse()?;
        let minimo: f64 = row[4].replace(',', ".").parse()?;
        let maximo: f64 = row[5].replace(',', ".").parse()?;
        let volume: f64 = row[6]
            .trim_end_matches(|c| c == 'M' || c == 'B')
            .replace(',', ".")
            .parse::<f64>()?
            * if row[6].ends_with('B') { 1_000.0 } else { 1.0 };

        inputs.push(vec![abertura, variacao, minimo, maximo, volume]);
        labels.push(fechamento);
    }

    // Normaliza os dados de treinamento
    let mut means = vec![0.0; 5];
    let mut stds = vec![0.0; 5];
    for feature in 0..5 {
        let mut column: Vec<f64> = inputs.iter().map(|row| row[feature]).collect();
        let (mean, std) = normalize(&mut column);
        means[feature] = mean;
        stds[feature] = std;
        for (i, value) in column.into_iter().enumerate() {
            inputs[i][feature] = value;
        }
    }

    // Divide os dados em treino e teste
    let split_idx = (inputs.len() as f64 * 0.8) as usize;
    let (train_inputs, test_inputs) = inputs.split_at(split_idx);
    let (train_labels, test_labels) = labels.split_at(split_idx);

    // Cria e treina o modelo MLP
    let mut model = rna::MLP::new(&[5, 64, 32, 16, 8, 1]); // Modelo mais profundo
    model.train(train_inputs, train_labels, 100, 0.001, "tanh"); // Usa tanh como ativação

    // Avalia o modelo nos dados de teste
    let mut test_loss = 0.0;
    for (input, label) in test_inputs.iter().zip(test_labels) {
        let prediction = model.forward(input, "tanh")[0]; // Usa tanh como ativação
        test_loss += (prediction - label).powi(2);
    }
    println!("Test Loss: {:.4}", test_loss / test_inputs.len() as f64);

    // Previsão para o dia 11/02/2025
    let input_11_fev_2025 = vec![
        54.21, // Abertura (valor do fechamento do dia anterior)
        0.50,  // Variação (hipótese: mesma variação do último dia)
        53.80, // Mínimo (mesmo valor mínimo do último dia)
        54.42, // Máximo (mesmo valor máximo do último dia)
        129.21, // Volume (em milhões, mesmo valor do último dia)
    ];

    // Normaliza os dados de entrada para previsão
    let normalized_input: Vec<f64> = input_11_fev_2025
        .iter()
        .enumerate()
        .map(|(i, &value)| (value - means[i]) / stds[i])
        .collect();

    // Faz a previsão
    let predicted_normalized = model.forward(&normalized_input, "tanh")[0];
    let predicted_denormalized = denormalize(predicted_normalized, means[0], stds[0]);

    println!(
        "Previsão de fechamento para 11/02/2025: {:.2}",
        predicted_denormalized
    );

    Ok(())
}