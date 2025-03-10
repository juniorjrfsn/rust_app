// main.rs
mod conexao;
mod mlp;

use thiserror::Error;
use std::error::Error as StdError;
use crate::conexao::read_file::ler_csv;
use crate::mlp::previsao::{treinar,prever}; // Import the previsao module

fn main() -> Result<(), Box<dyn StdError>> {
    let cotac_fonte = "investing";
    let ativo_financeiro = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo_financeiro); // Path to the CSV file
    let matrix = ler_csv(&file_path, cotac_fonte, ativo_financeiro)?;
    let (model, means, stds, label_mean, label_std) = treinar(matrix.clone())?;
    prever(
        matrix,
        model,
        &means,
        &stds,
        label_mean,
        label_std,
        cotac_fonte,
        ativo_financeiro,
    )?;
    Ok(())
}