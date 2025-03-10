// src/main.rs
mod conexao;
mod mlp;

use std::error::Error as StdError;
use burn::tensor::backend::NdArrayBackend;
use crate::conexao::read_file::ler_csv;
use crate::mlp::previsao_lstm::treinar;

fn main() -> Result<(), Box<dyn StdError>> {
    let device = NdArrayBackend::default().device();
    let cotac_fonte = "investing";
    let ativo = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo);
    
    let matrix = ler_csv(&file_path, cotac_fonte)?;
    treinar::<NdArrayBackend<f32>>(matrix, &device)?;
    
    Ok(())
}