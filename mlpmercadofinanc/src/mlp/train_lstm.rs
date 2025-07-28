// File : src/mlp/train_lstm.rs



use burn::{
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Lstm, LstmConfig},
    optim::AdamConfig,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{
        metric::{LossMetric, LearningRateMetric},
        TrainStep, TrainOutput,
        LearnerBuilder,
    },
    data::dataloader::{DataLoaderBuilder, Dataset, batcher::Batcher},
    record::{Recorder, BinFileRecorder, FullPrecisionSettings},
};
use serde::{Serialize, Deserialize};
use chrono::NaiveDate;
use crate::utils::AppError;

#[derive(Module, Debug)]
pub struct LSTMModel<B: AutodiffBackend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            input_size: 5,
            hidden_size: 64,
            learning_rate: 0.001,
            num_epochs: 50,
            batch_size: 32,
        }
    }
}

impl<B: AutodiffBackend> LSTMModel<B> {
    pub fn new(config: &LSTMConfig, device: &B::Device) -> Self {
        let lstm = LstmConfig::new(config.input_size, config.hidden_size, false).init(device);
        let linear = LinearConfig::new(config.hidden_size, 1).init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        let batch_size = outputs.dims()[0];
        let seq_len = outputs.dims()[1];
        let hidden_size = outputs.dims()[2];
        
        let last_output = outputs.clone().slice([0..batch_size, seq_len-1..seq_len]);
        let last_output_2d = last_output.reshape([batch_size, hidden_size]);
        self.linear.forward(last_output_2d)
    }
}

#[derive(Clone, Debug)]
pub struct LSTMDataset<B: AutodiffBackend> {
    inputs: Tensor<B, 3>,
    targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct LSTMItem<B: AutodiffBackend> {
    pub input: Tensor<B, 3>,
    pub target: Tensor<B, 2>,
}

impl<B: AutodiffBackend> Dataset<LSTMItem<B>> for LSTMDataset<B> {
    fn get(&self, index: usize) -> Option<LSTMItem<B>> {
        if index >= self.len() {
            return None;
        }
        let input = self.inputs.clone().slice([index..index+1]);
        let target = self.targets.clone().slice([index..index+1]);
        Some(LSTMItem { input, target })
    }

    fn len(&self) -> usize {
        self.inputs.dims()[0]
    }
}

#[derive(Clone)]
pub struct LSTMItemBatcher<B: AutodiffBackend> {
    device: B::Device,
}

impl<B: AutodiffBackend> LSTMItemBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: AutodiffBackend> Batcher<LSTMItem<B>, LSTMItem<B>, B> for LSTMItemBatcher<B> {
    fn batch(&self, items: Vec<LSTMItem<B>>, device: &B::Device) -> LSTMItem<B> {
        let inputs = items.iter().map(|item| item.input.clone()).collect::<Vec<_>>();
        let targets = items.iter().map(|item| item.target.clone()).collect::<Vec<_>>();
        
        let input = Tensor::cat(inputs, 0).to_device(device);
        let target = Tensor::cat(targets, 0).to_device(device);
        
        LSTMItem { input, target }
    }
}

pub struct LSTMTrainStep<B: AutodiffBackend> {
    model: LSTMModel<B>,
    optimizer: AdamConfig,
}

impl<B: AutodiffBackend> TrainStep<LSTMItem<B>, LSTMItem<B>> for LSTMTrainStep<B>
where
    B::InnerBackend: AutodiffBackend,
{
    fn step(&self, item: LSTMItem<B>) -> TrainOutput<LSTMItem<B>> {
        let output = self.model.forward(item.input.clone());
        let loss = MseLoss::new().forward(output.clone(), item.target.clone(), burn::nn::loss::Reduction::Mean);
        
        TrainOutput::new(&self.model, loss.backward(), LSTMItem {
            input: item.input,
            target: output,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub target_mean: f32,
    pub target_std: f32,
    pub feature_means: Vec<f32>,
    pub feature_stds: Vec<f32>,
}

pub fn train<B: AutodiffBackend>(
    x: Tensor<B, 3>,
    y: Tensor<B, 2>,
    dates: Vec<NaiveDate>,
    target_mean: f32,
    target_std: f32,
    feature_means: [f32; 5],
    feature_stds: [f32; 5],
    device: &B::Device,
    model_path: &str,
) -> Result<(), AppError>
where
    B::InnerBackend: burn::record::Recorder<B::InnerBackend>,
{
    let config = LSTMConfig::default();
    let model = LSTMModel::<B>::new(&config, device);
    let mut optimizer = AdamConfig::new();
    optimizer.learning_rate = config.learning_rate;
    let loss_fn = MseLoss::new();

    let dataset = LSTMDataset { inputs: x, targets: y };
    let split_date = chrono::NaiveDate::from_ymd_opt(2023, 1, 1)
        .ok_or_else(|| AppError::InvalidDate("Invalid split date".into()))?;
    let split_idx = dates.iter().position(|d| *d >= split_date)
        .unwrap_or((dataset.len() as f32 * 0.8) as usize);

    let train_dataset = burn::data::dataset::InMemDataset::new(
        (0..split_idx).filter_map(|i| dataset.get(i)).collect::<Vec<_>>()
    );
    let valid_dataset = burn::data::dataset::InMemDataset::new(
        (split_idx..dataset.len()).filter_map(|i| dataset.get(i)).collect::<Vec<_>>()
    );

    let batcher = LSTMItemBatcher::new(device.clone());
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(1234)
        .num_workers(4)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(valid_dataset);

    let learner = LearnerBuilder::new(model_path)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .build(model, optimizer);

    let trained_model = learner.fit(train_dataloader, valid_dataloader);

    let metadata = ModelMetadata {
        target_mean,
        target_std,
        feature_means: feature_means.to_vec(),
        feature_stds: feature_stds.to_vec(),
    };
    std::fs::write(
        format!("{}.metadata.json", model_path),
        serde_json::to_string(&metadata)?,
    )?;

    BinFileRecorder::<FullPrecisionSettings>::new()
        .save_item(trained_model, model_path.into())?;
    Ok(())
}