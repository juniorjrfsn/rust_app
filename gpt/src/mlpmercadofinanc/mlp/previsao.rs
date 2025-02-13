pub mod prev_cotacao {
    use std::error::Error; // Importa o trait Error
    use chrono::Local; // Importa a função Local
    use crate::mlp::mlp_cotacao::rna;
    use crate::conexao::db;
 

    // Função para desnormalizar os dados
    fn denormalize(value: f64, mean: f64, std: f64) -> f64 {
        value * std + mean
    }
    // Função para normalizar os dados
    fn normalize(data: &mut [f64]) -> (f64, f64) {
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let std: f64 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        for x in data {
            *x = (*x - mean) / std;
        }
        (mean, std)
    }

      // Função para treinar o modelo
    pub fn treinar(matrix: Vec<Vec<String>>) -> Result<(rna::MLP, Vec<f64>, Vec<f64>, f64, f64), Box<dyn Error>> {
        // Função auxiliar para converter uma linha de dados em formato numérico
        fn parse_row(row: &[String]) -> Option<Vec<f64>> {
            if row[1] == "n/d" || row[2] == "n/d" {
                return None; // Ignora linhas inválidas
            }
            let abertura: f64 = row[1].replace(',', ".").parse().ok()?;
            let _fechamento: f64 = row[2].replace(',', ".").parse().ok()?;
            let variacao: f64 = row[3].replace(',', ".").replace('%', "").parse().ok()?;
            let minimo: f64 = row[4].replace(',', ".").parse().ok()?;
            let maximo: f64 = row[5].replace(',', ".").parse().ok()?;
            let volume: f64 = row[6]
                .trim_end_matches(|c| c == 'M' || c == 'B')
                .replace(',', ".")
                .parse::<f64>()
                .ok()?
                * if row[6].ends_with('B') { 1_000.0 } else { 1.0 };
            Some(vec![abertura, variacao, minimo, maximo, volume])
        }
    
        // Converte os dados para formato numérico
        let mut inputs = Vec::new();
        let mut labels = Vec::new();
        for row in matrix.iter().skip(1) {
            if let Some(parsed_row) = parse_row(row) {
                inputs.push(parsed_row);
                labels.push(row[2].replace(',', ".").parse::<f64>().unwrap());
            }
        }
    
        // Função auxiliar para normalizar uma coluna de dados
        fn normalize_column(column: &mut [f64]) -> (f64, f64) {
            let mean: f64 = column.iter().sum::<f64>() / column.len() as f64;
            let std: f64 = (column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64).sqrt();
            for x in column {
                *x = (*x - mean) / std;
            }
            (mean, std)
        }
    
        // Normaliza os dados de treinamento
        let mut means = vec![0.0; 5];
        let mut stds = vec![0.0; 5];
        for feature in 0..5 {
            let column: Vec<f64> = inputs.iter().map(|row| row[feature]).collect();
            let (mean, std) = normalize_column(&mut column.clone());
            means[feature] = mean;
            stds[feature] = std;
            for (input, value) in inputs.iter_mut().zip(column) {
                input[feature] = value;
            }
        }
    
        // Calcula a média e o desvio padrão dos rótulos
        let label_mean: f64 = labels.iter().sum::<f64>() / labels.len() as f64;
        let label_std: f64 = (labels.iter().map(|x| (x - label_mean).powi(2)).sum::<f64>() / labels.len() as f64).sqrt();
    
        // Divide os dados em treino e teste
        let split_idx = (inputs.len() as f64 * 0.8) as usize;
        let (train_inputs, test_inputs) = inputs.split_at(split_idx);
        let (train_labels, test_labels) = labels.split_at(split_idx);
    
        // Cria e treina o modelo MLP
        const ARCHITECTURE: &[usize] = &[5, 64, 32, 16, 8, 1];
        const EPOCHS: usize = 100;
        const LEARNING_RATE: f64 = 0.001;
        const ACTIVATION: &str = "tanh";
    
        let mut model = rna::MLP::new(ARCHITECTURE);
        model.train(train_inputs, train_labels, EPOCHS, LEARNING_RATE, ACTIVATION);
    
        // Avalia o modelo nos dados de teste
        let mut test_loss = 0.0;
        for (input, label) in test_inputs.iter().zip(test_labels) {
            let prediction = model.forward(input, ACTIVATION)[0];
            test_loss += (prediction - label).powi(2);
        }
        println!("Test Loss: {:.4}", test_loss / test_inputs.len() as f64);

        // Salva o modelo treinado e os parâmetros de normalização no banco de dados
        let conn = db::init_db()?;
        let serialized_model = model.serialize()?;
        db::insert_modelo(&conn, "WEGE3", &serialized_model, &means, &stds, label_mean, label_std)?;

        println!("Modelo treinado e salvo no banco de dados.");
        Ok((model, means, stds, label_mean, label_std))
        // Ok(())
    }

    // Função para prever o valor de fechamento para o próximo dia
    pub fn prever(matrix: Vec<Vec<String>>, cotac_fonte: &str, ativo_financeiro: &str) -> Result<(), Box<dyn Error>> {
        let conn = db::init_db()?;
    
        // Recupera o modelo treinado e os parâmetros de normalização do banco de dados
        let (serialized_model, means, stds, label_mean, label_std) = if let Some(data) = db::get_modelo(&conn, ativo_financeiro)? {
            data
        } else {
            return Err("Modelo não encontrado no banco de dados".into());
        };
    
        // Desserializa o modelo
        let model = rna::MLP::deserialize(&serialized_model)?;
    
        // Pega os valores da última linha (último registro disponível)
        let ultima_linha = &matrix[matrix.len() - 1];
        let abertura: f64 = ultima_linha[1].replace(',', ".").parse()?;
        let fechamento: f64 = ultima_linha[2].replace(',', ".").parse()?;
        let variacao: f64 = ultima_linha[3].replace(',', ".").replace('%', "").parse()?;
        let minimo: f64 = ultima_linha[4].replace(',', ".").parse()?;
        let maximo: f64 = ultima_linha[5].replace(',', ".").parse()?;
        let volume: f64 = ultima_linha[6]
            .trim_end_matches(|c| c == 'M' || c == 'B')
            .replace(',', ".")
            .parse::<f64>()?
            * if ultima_linha[6].ends_with('B') { 1_000.0 } else { 1.0 };
    
        // Cria o vetor de previsão para amanhã com os valores da última linha 
        let previsao_para_amanha = vec![abertura, variacao, minimo, maximo, volume];

        // Normaliza os dados de entrada para previsão
        let normalized_input: Vec<f64> = previsao_para_amanha
            .iter()
            .enumerate()
            .map(|(i, &value)| (value - means[i]) / stds[i])
            .collect();
    
        // Faz a previsão
        let predicted_normalized = model.forward(&normalized_input, "tanh")[0];
        let predicted_denormalized = denormalize(predicted_normalized, label_mean, label_std);
    
        // Exibe os resultados
        println!("Dados do último registro do CSV:");
        println!(
            "Abertura: {:.2}, Variação: {:.2}%, Mínimo: {:.2}, Máximo: {:.2}, Volume: {:.2}, Fechamento: {:.2}",
            abertura, variacao, minimo, maximo, volume, fechamento
        );
        println!();
        println!(
            "Ativo: {} - Previsão de fechamento para amanhã: {:.2}",
            ativo_financeiro, predicted_denormalized
        );
    
        // Armazena a previsão no banco de dados
        let data_prevista = chrono::Local::now().format("%Y-%m-%d").to_string(); // Data atual como exemplo
        db::insert_previsao(&conn, ativo_financeiro, &data_prevista, predicted_denormalized)?;
    
        Ok(())
    }
}