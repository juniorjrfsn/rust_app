use std::error::Error;
use thiserror::Error;

mod conexao;
use crate::conexao::read_file::ler_csv;
use crate::conexao::db; // Importa o módulo db

mod mlp;
// use crate::mlp::mlp_cotacao::rna;
use crate::mlp::previsao::prev_cotacao;

// Example in read_file.rs:
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    // ... other errors
}

// Example in main.rs:
use thiserror::Error;
 


fn main() -> Result<(), Box<dyn Error>> {
    // Define a fonte de cotação e o ativo financeiro
    let cotac_fonte = "investing";
    let ativo_financeiro = "WEGE3";

    // Constrói o caminho do arquivo CSV
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo_financeiro);

    // Carrega os dados do CSV
    let matrix = ler_csv(&file_path, &cotac_fonte, &ativo_financeiro)?;

    // Treina o modelo e obtém os parâmetros de normalização
    //let matrix_clone = matrix.clone(); // Clona os dados para evitar consumo
    //let (model, means, stds, label_mean, label_std) = prev_cotacao::treinar(matrix_clone)?;

    // Treina o modelo se ele ainda não estiver no banco de dados
    let conn = db::init_db()?;
    if db::get_modelo(&conn, ativo_financeiro)?.is_none() {
        prev_cotacao::treinar(matrix.clone())?;
    }

    // Faz a previsão usando o modelo carregado do banco de dados
    prev_cotacao::prever(matrix, &cotac_fonte, &ativo_financeiro)?;

    Ok(())
}