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
    let ativo_financeiro = "WEGE3";
    let file_path = format!("dados/{}.csv", ativo_financeiro);
    let matrix = ler_csv(&file_path)?;

    // Verifica se há pelo menos uma linha de dados após o cabeçalho
    if matrix.len() < 2 {
        return Err("O arquivo CSV não contém dados suficientes.".into());
    }

    // Pega os valores da segunda linha (primeira linha de dados após o cabeçalho)
    let segunda_linha = &matrix[0]; // Index 0 porque o cabeçalho é retirado no retorno da função ler_csv

    // Converte os valores da segunda linha para o formato numérico
    let _abertura: f64 = segunda_linha[1].replace(',', ".").parse()?;
    let _fechamento: f64 = segunda_linha[2].replace(',', ".").parse()?;
    let _variacao: f64 = segunda_linha[3].replace(',', ".").parse()?;
    let _minimo: f64 = segunda_linha[4].replace(',', ".").parse()?;
    let _maximo: f64 = segunda_linha[5].replace(',', ".").parse()?;
    let _volume: f64 = segunda_linha[6]
        .trim_end_matches(|c| c == 'M' || c == 'B')
        .replace(',', ".")
        .parse::<f64>()?
        * if segunda_linha[6].ends_with('B') { 1_000.0 } else { 1.0 };

    // Cria o vetor de previsão para amanhã com os valores da segunda linha
    let previsao_para_amanha = vec![_abertura, _variacao, _minimo, _maximo, _volume];

    // Converte os dados para formato numérico (restante do CSV)
    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    for row in matrix.iter().skip(1) { // Pula o cabeçalho
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

    // Normaliza os dados de entrada para previsão
    let normalized_input: Vec<f64> = previsao_para_amanha
        .iter()
        .enumerate()
        .map(|(i, &value)| (value - means[i]) / stds[i])
        .collect();

    // Faz a previsão
    let predicted_normalized = model.forward(&normalized_input, "tanh")[0];
    let predicted_denormalized = denormalize(predicted_normalized, means[0], stds[0]);

    println!();
    println!("Dados do primeiro registro do CSV (Segunda linha que é contém os últimos valores da cotação):");
    println!(
        "abertura: {} - variacao: {} - minimo: {} - maximo: {} - volume: {} - fechamento: {}",
        _abertura, _variacao, _minimo, _maximo, _volume, _fechamento
    );
    println!();
    println!(
        "Ativo: {} - Previsão de fechamento para amanhã: {:.2}",
        ativo_financeiro, predicted_denormalized
    );

    Ok(())
}