// File : src/main.rs


use burn::{
    backend::Autodiff,
    config::Config,
    data::dataloader::{DataLoaderBuilder, Dataset},
    module::Module,
    optim::AdamConfig,
    tensor::{backend::Backend, Tensor},
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep},
};
use log::{error, info};
use crate::mlp::common::{AppError, load_normalization_params, parse_row, preprocess, save_normalization_params};
use crate::mlp::model::LSTMModel;
use crate::conexao::read_file::ler_csv;

#[derive(Config)]
struct LSTMConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
}

impl LSTMConfig {
    pub fn init() -> Self {
        Self {
            input_size: 5,
            hidden_size: 64,
            sequence_length: 30,
            learning_rate: 0.001,
            num_epochs: 100,
        }
    }
}

#[derive(Clone)]
struct FinancialDataset {
    sequences: Vec<Vec<Vec<f32>>>,
    targets: Vec<f32>,
}

impl Dataset<f32> for FinancialDataset {
    fn get(&self, index: usize) -> Option<(Vec<Vec<f32>>, f32)> {
        self.sequences.get(index).map(|seq| (seq.clone(), self.targets[index]))
    }

    fn len(&self) -> usize {
        self.sequences.len()
    }
}

struct LSTMTraining<B: Backend> {
    model: LSTMModel<B>,
    optimizer: burn::optim::Adam<B>,
}

impl<B: Backend> LSTMTraining<B> {
    pub fn new(model: LSTMModel<B>, optimizer: burn::optim::Adam<B>) -> Self {
        Self { model, optimizer }
    }
}

impl<B: Backend> TrainStep<(Tensor<B, 3>, Tensor<B, 2>), Tensor<B, 2>> for LSTMTraining<B> {
    fn step(&self, (inputs, targets): (Tensor<B, 3>, Tensor<B, 2)) -> TrainOutput<Tensor<B, 2>> {
        let output = self.model.forward(inputs);
        let loss = burn::nn::loss::mse_loss(output.clone(), targets);
        TrainOutput::new(&self.model, loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<(Tensor<B, 3>, Tensor<B, 2>), Tensor<B, 2>> for LSTMTraining<B> {
    fn step(&self, (inputs, targets): (Tensor<B, 3>, Tensor<B, 2)) -> Tensor<B, 2> {
        self.model.forward(inputs)
    }
}

fn train<B: Backend>(
    file_path: &str,
    cotac_fonte: &str,
    config: &LSTMConfig,
    device: &B::Device,
    model_path: &str,
    norm_path: &str,
) -> Result<(), AppError> {
    let matrix = ler_csv(file_path, cotac_fonte)?;
    let (x, y, _, target_mean, target_std) = preprocess::<B>(&matrix, config.sequence_length, device)?;
    info!("Prepared {} training sequences", x.dims()[0]);

    let dataset = FinancialDataset {
        sequences: x.to_data().value.into_iter()
            .collect::<Vec<_>>()
            .chunks(config.sequence_length * config.input_size)
            .map(|chunk| chunk.chunks(config.input_size).map(|c| c.to_vec()).collect())
            .collect(),
        targets: y.to_data().value,
    };

    let train_size = (dataset.len() as f64 * 0.8) as usize;
    let train_dataset = FinancialDataset {
        sequences: dataset.sequences[..train_size].to_vec(),
        targets: dataset.targets[..train_size].to_vec(),
    };
    let valid_dataset = FinancialDataset {
        sequences: dataset.sequences[train_size..].to_vec(),
        targets: dataset.targets[train_size..].to_vec(),
    };

    let model = LSTMModel::new(config.input_size, config.hidden_size, device);
    let optimizer = AdamConfig::new().init();
    let learner = LearnerBuilder::new(".")
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .with_file_checkpointer(model_path)
        .build(model, optimizer, config.learning_rate);

    learner.fit(
        DataLoaderBuilder::new(train_dataset)
            .batch_size(32)
            .build(),
        DataLoaderBuilder::new(valid_dataset)
            .batch_size(32)
            .build(),
    );

    save_normalization_params(target_mean, target_std, norm_path)?;
    info!("Model saved to {}, normalization params saved to {}", model_path, norm_path);
    Ok(())
}

fn predict<B: Backend>(
    file_path: &str,
    cotac_fonte: &str,
    config: &LSTMConfig,
    device: &B::Device,
    model_path: &str,
    norm_path: &str,
) -> Result<Vec<(String, f32)>, AppError> {
    let matrix = ler_csv(file_path, cotac_fonte)?;
    let (x, _, dates, _, _) = preprocess::<B>(&matrix, config.sequence_length, device)?;
    let (target_mean, target_std) = load_normalization_params(norm_path)?;
    let mut model = LSTMModel::new(config.input_size, config.hidden_size, device);
    model.load(model_path, device)?;
    let output = model.forward(x);
    let output_data = output.to_data();
    if output_data.shape.dims != [dates.len(), 1] {
        return Err(AppError::InvalidData {
            location: "output".to_string(),
            message: format!("Unexpected output shape: {:?}", output_data.shape),
        });
    }
    let predictions: Vec<f32> = output_data.value.into_iter().map(|x| x * target_std + target_mean).collect();
    let results: Vec<(String, f32)> = dates.into_iter().zip(predictions).collect();

    if let Some((last_date, last_pred)) = results.last() {
        let last_row = matrix.last().ok_or_else(|| AppError::InvalidData {
            location: "matrix".to_string(),
            message: "Empty matrix".to_string(),
        })?;
        let parsed_row = parse_row(last_row)?;
        println!(
            "Last CSV record: Date: {}, Open: {:.2}, High: {:.2}, Low: {:.2}, Variation: {:.2}%, Volume: {:.2}",
            last_date, parsed_row[0], parsed_row[1], parsed_row[2], parsed_row[3], parsed_row[4]
        );
        println!("Predicted closing price for {}: {:.2}", last_date, last_pred);
    }

    info!("Generated {} predictions", results.len());
    Ok(results)
}

fn main() {
    env_logger::init();
    let config = LSTMConfig::init();
    let device = burn::backend::WgpuDevice::default();
    let file_path = "data.csv"; // Replace with actual CSV path
    let model_path = "lstm_model";
    let norm_path = "norm_params.json";
    let cotac_fonte = "investing"; // or "infomoney"

    if let Err(e) = train::<Autodiff<burn::backend::Wgpu>>(file_path, cotac_fonte, &config, &device, model_path, norm_path) {
        error!("Training failed: {}", e);
        return;
    }

    if let Err(e) = predict::<burn::backend::Wgpu>(file_path, cotac_fonte, &config, &device, model_path, norm_path) {
        error!("Prediction failed: {}", e);
    }
}





// cargo run --bin mlpmercadofinanc -- --phase treino --model-path lstm_model.burn

//  cargo run --bin mlpmercadofinanc -- --phase previsao --model-path lstm_model.burn

//
