// File : src/main.rs 


mod conexao;
mod mlp;

use std::error::Error as StdError;
use clap::{Arg, Command};
// Import the correct backend types from burn::backend
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

    // Define the backend types correctly for Burn 0.18
    // NdArrayBackend<f32> for basic operations, Autodiff<NdArrayBackend<f32>> for training
    type Backend = NdArrayBackend<f32>;
    type MyAutodiffBackend = Autodiff<Backend>;
    
    // Create the device instance correctly
    let device = Backend::default(); // This creates the default device (CPU for NdArray)
    let cotac_fonte = "investing";
    let ativo = "WEGE3";
    let file_path = format!("dados/{}/{}.csv", cotac_fonte, ativo);
    let matrix = crate::conexao::read_file::ler_csv(&file_path, cotac_fonte, ativo)?;
    
    // Call the training function with the correct Autodiff backend type
    crate::mlp::train_lstm::train::<MyAutodiffBackend>(matrix, &device)?; // Pass matrix by value
    Ok(())
}




// cargo run --bin mlpmercadofinanc -- --phase treino --model-path lstm_model.burn

//  cargo run --bin mlpmercadofinanc -- --phase previsao --model-path lstm_model.burn

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type rf --model-path rf_model.bin --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type rf --model-path rf_model.bin --ativo WEGE3 --fonte investing



// cargo run --bin mlpmercadofinanc -- --phase previsao --model-type mlp --model-path mlp_model.bin --ativo WEGE3 --fonte investing


// cargo run --bin mlpmercadofinanc -- --phase treino --model-type mlp --model-path mlp_model.bin --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase previsao --model-type lstm --model-path lstm_model.burn --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type lstm --model-path lstm_model.burn --ativo WEGE3 --fonte investing

// cargo run -- --phase predict --asset WEGE3 --source investing

// cargo run -- --phase train --asset WEGE3 --source investing --model-path model.burn

// cargo run -- --phase train --asset WEGE3 --source investing --model-path model.burn