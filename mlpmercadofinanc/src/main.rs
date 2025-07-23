// File : src/main.rs


mod conexao;
mod mlp;

use std::error::Error as StdError;
use clap::{Arg, Command};
// Import the correct backend types
use burn::backend::{NdArrayBackend, Autodiff};

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

    // Define the backend types correctly
    type Backend = NdArrayBackend<f32>;
    type MyAutodiffBackend = Autodiff<Backend>; // Use the imported Autodiff
    
    let device = Backend::default(); // Create device instance correctly
    let cotac_fonte = "investing";
    let ativo = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo);
    let matrix = crate::conexao::read_file::ler_csv(&file_path, cotac_fonte, ativo)?;
    
    // Call training function with the correct Autodiff backend
    crate::mlp::train_lstm::treinar::<MyAutodiffBackend>(matrix, &device)?;
    Ok(())
}



// cargo run --bin mlpmercadofinanc -- --ph treino

// 