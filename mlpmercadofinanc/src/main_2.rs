// File : src/main.rs


use burn::backend::{NdArray, Autodiff}; // Updated import
use std::error::Error as StdError;
use clap::{Arg, Command};
use crate::mlp::train_lstm;
use crate::mlp::previsao_lstm;

mod conexao;
mod mlp;

fn main() -> Result<(), Box<dyn StdError>> {
    let matches = Command::new("mlpmercadofinanc")
        .arg(Arg::new("phase")
            .short('p')
            .long("phase")
            .value_name("PHASE")
            .required(true)
            .help("Phase: 'treino' or 'previsao'"))
        .arg(Arg::new("model_path")
            .short('m')
            .long("model-path")
            .value_name("MODEL_PATH")
            .default_value("lstm_model.burn")
            .help("Path to save/load the model"))
        .get_matches();

    let phase = matches.get_one::<String>("phase").unwrap();
    let model_path = matches.get_one::<String>("model_path").unwrap();
    type Backend = NdArray<f32>;
    type MyAutodiffBackend = Autodiff<Backend>;

    let device = Backend::Device::default();
    let cotac_fonte = "investing";
    let ativo = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo);
    let matrix = crate::conexao::read_file::ler_csv(&file_path, cotac_fonte, ativo)?;

    match phase.as_str() {
        "treino" => {
            train_lstm::treinar::<MyAutodiffBackend>(matrix, &device, model_path)?;
        }
        "previsao" => {
            let predictions = previsao_lstm::predict::<Backend>(&matrix, &device, model_path)?;
            println!("Previsões para {} (preço de fechamento): {:?}", ativo, predictions);
        }
        _ => return Err("Phase must be 'treino' or 'previsao'".into()),
    }

    Ok(())
}



// cargo run --bin mlpmercadofinanc -- --phase treino --model-path lstm_model.burn

//  cargo run --bin mlpmercadofinanc -- --phase previsao --model-path lstm_model.burn

//