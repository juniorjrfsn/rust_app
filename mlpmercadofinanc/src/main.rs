// src/main.rs
mod conexao;
mod mlp;

use std::error::Error as StdError;
use burn::tensor::backend::NdArrayBackend;
use crate::conexao::read_file::ler_csv;
use crate::mlp::previsao_lstm::treinar;
use clap::{Arg, Command};
use burn::backend::NdArrayBackend;
use burn::backend::NdArray;
use crate::mlp::train_lstm::treinar;

fn main() -> Result<(), Box<dyn StdError>> {
    let matches = Command::new("mlpmercadofinanc")
        .arg(Arg::new("phase")
            .short('p')
            .long("phase")
            .value_name("PHASE")
            .required(true))
        .get_matches();

    let phase = matches.get_one::<String>("phase").unwrap();
    if phase != "treino" {
        return Err("Only 'treino' phase is supported".into());
    }

    let device = NdArray::Device::default();
    let cotac_fonte = "investing";
    let ativo = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo);
    let matrix = ler_csv(&file_path, cotac_fonte, ativo)?;
    treinar::<NdArray>(&matrix, &device)?;
    Ok(())
}
 
// cargo run --bin mlpmercadofinanc -- --ph treino

// 