// File : src/main.rs 
 


use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use crate::conexao::read_file::{read_raw_data, ler_csv};
use crate::mlp::train_lstm::train;
use crate::mlp::predict_lstm::{predict_single, predict_batch};
use crate::utils::AppError;

mod conexao;
mod mlp;
mod utils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    phase: String,
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long)]
    model_path: String,
    #[arg(long, default_value_t = 10)]
    seq_length: usize,
}

fn main() -> Result<(), AppError> {
    let args = Args::parse();
    let file_path = format!("dados/{}/{}.csv", args.asset, args.source);
 
    type MyBackend = Autodiff<Wgpu>;
    type MyInferenceBackend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    match args.phase.as_str() {
        "train" => {
            let (x, y, dates, target_mean, target_std, feature_means, feature_stds) = 
                ler_csv::<MyBackend>(&file_path, args.seq_length, &device)?;
            train::<MyBackend>(
                x, y, dates, target_mean, target_std, feature_means, feature_stds, 
                &device, &args.model_path
            )?;
            println!("Model trained and saved to {}", args.model_path);
        }
        "predict" => {
            let records = read_raw_data(&file_path)?;
            let (date, prediction) = predict_single::<MyInferenceBackend>(
                &records, args.seq_length, &device, &args.model_path
            )?;
            println!("Prediction for {} on {}: {:.2}", args.asset, date, prediction);

            let predictions = predict_batch::<MyInferenceBackend>(
                &records, args.seq_length, &device, &args.model_path
            )?;
            println!("Batch predictions for {}:", args.asset);
            for (date, pred) in predictions {
                println!("  {}: {:.2}", date, pred);
            }
        }
        _ => return Err(AppError::InvalidData(format!("Unknown phase: {}", args.phase))),
    }

    Ok(())
}

// cargo run --bin mlpmercadofinanc -- --phase treino --model-path lstm_model.burn

//  cargo run --bin mlpmercadofinanc -- --phase previsao --model-path lstm_model.burn

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type rf --model-path rf_model.bin --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type rf --model-path rf_model.bin --ativo WEGE3 --fonte investing



// cargo run --bin mlpmercadofinanc -- --phase previsao --model-type mlp --model-path mlp_model.bin --ativo WEGE3 --fonte investing


// cargo run --bin mlpmercadofinanc -- --phase treino --model-type mlp --model-path mlp_model.bin --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase previsao --model-type lstm --model-path lstm_model.burn --ativo WEGE3 --fonte investing



// cargo run -- --phase predict --asset WEGE3 --source investing

 

// cargo run --bin mlpmercadofinanc -- --phase treino --model-type lstm --model-path lstm_model.burn --ativo WEGE3 --fonte investing

// cargo run --bin mlpmercadofinanc -- --phase predict --model-type lstm --model-path lstm_model.burn --ativo WEGE3 --fonte investing

 

// cargo run -- --phase train --asset WEGE3 --source investing --model-path model.burn

// cargo run -- --phase predict --asset WEGE3 --source investing --model-path model.burn

 
 
 

// cargo run -- --phase train --asset WEGE3 --source investing --model-path lstm_model.burn --seq-length 10

// cargo run -- --phase predict --asset WEGE3 --source investing --model-path lstm_model.burn --seq-length 10
