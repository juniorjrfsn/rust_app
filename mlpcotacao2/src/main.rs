mod conexao;
mod mlp;

use std::fs;
use std::error::Error as StdError;
use crate::conexao::read_file::ler_csv;
use crate::mlp::previsao::{treinar, prever};

fn main() -> Result<(), Box<dyn StdError>> {
    // Configurações iniciais
    let cotac_fonte = "infomoney"; // Fonte dos dados ("investing" ou "infomoney")
    let ativo_financeiro = "SLCE3"; // Ativo financeiro (ex.: "WEGE3", "SLCE3")
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo_financeiro); // Caminho do arquivo CSV

    // Verifica se o arquivo existe
    if !fs::metadata(&file_path).is_ok() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Arquivo não encontrado: {}", file_path),
        )));
    }

    // Lê os dados do CSV
    println!("Lendo dados do arquivo: {}", file_path);
    let matrix = match ler_csv(&file_path, cotac_fonte, ativo_financeiro) {
        Ok(matrix) => {
            println!("Dados parseados com sucesso.");
            matrix
        }
        Err(e) => {
            eprintln!("Erro ao ler o arquivo CSV: {:?}", e);
            return Err(Box::new(e));
        }
    };

    // Treina o modelo
    println!("Treinando o modelo...");
    let train_ratio = 0.8; // 80% dos dados para treino
    let (model, means, stds, label_mean, label_std) = match treinar(matrix.clone(), train_ratio) {
        Ok(result) => {
            println!("Modelo treinado com sucesso.");
            result
        }
        Err(e) => {
            eprintln!("Erro ao treinar o modelo: {:?}", e);
            return Err(Box::new(e));
        }
    };

    // Faz a previsão
    println!("Fazendo previsão...");
    match prever(
        matrix,
        model,
        &means,
        &stds,
        label_mean,
        label_std,
        ativo_financeiro,
    ) {
        Ok(_) => println!("Previsão concluída com sucesso."),
        Err(e) => {
            eprintln!("Erro ao fazer a previsão: {:?}", e);
            return Err(Box::new(e));
        }
    };

    Ok(())
}