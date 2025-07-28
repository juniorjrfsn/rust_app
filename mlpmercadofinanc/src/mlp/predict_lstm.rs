// file : src/mlp/predict_lstm.rs



use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
    record::{Recorder, BinFileRecorder, FullPrecisionSettings},
};
use chrono::NaiveDate;
use crate::mlp::train_lstm::{LSTMModel, LSTMConfig, ModelMetadata};
use crate::utils::{AppError, Record, parse_row};
use crate::mlp::data_utils::preprocess_lstm_single;

pub fn predict_single<B: Backend>(
    records: &[Record],
    seq_length: usize,
    device: &B::Device,
    model_path: &str,
) -> Result<(NaiveDate, f32), AppError>
where
    B: burn::record::Recorder<B>,
{
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(format!("{}.metadata.json", model_path))?,
    )?;

    let (x, date) = preprocess_lstm_single(
        records,
        seq_length,
        device,
        &metadata.feature_means.try_into().map_err(|_| AppError::InvalidData("Invalid feature means".into()))?,
        &metadata.feature_stds.try_into().map_err(|_| AppError::InvalidData("Invalid feature stds".into()))?,
    )?;

    let config = LSTMConfig::default();
    let model = LSTMModel::<B>::new(&config, device);
    let mut model_path_buf = model_path.into();
    let model = BinFileRecorder::<FullPrecisionSettings>::new()
        .load_item(&mut model_path_buf)?.init();

    let output = model.forward(x);
    if output.dims() != [1, 1] {
        return Err(AppError::InvalidData("Unexpected output shape".into()));
    }

    let prediction = output.into_data().value[0];
    let denormalized_prediction = prediction * metadata.target_std + metadata.target_mean;

    Ok((date, denormalized_prediction))
}

pub fn predict_batch<B: Backend>(
    records: &[Record],
    seq_length: usize,
    device: &B::Device,
    model_path: &str,
) -> Result<Vec<(NaiveDate, f32)>, AppError>
where
    B: burn::record::Recorder<B>,
{
    let metadata: ModelMetadata = serde_json::from_str(
        &std::fs::read_to_string(format!("{}.metadata.json", model_path))?,
    )?;

    let (x, dates) = preprocess_batch::<B>(
        records,
        seq_length,
        device,
        &metadata.feature_means,
        &metadata.feature_stds,
    )?;

    let config = LSTMConfig::default();
    let model = LSTMModel::<B>::new(&config, device);
    let mut model_path_buf = model_path.into();
    let model = BinFileRecorder::<FullPrecisionSettings>::new()
        .load_item(&mut model_path_buf)?.init();

    let output = model.forward(x);
    if output.dims()[0] != dates.len() || output.dims()[1] != 1 {
        return Err(AppError::InvalidData("Unexpected output shape".into()));
    }

    let predictions: Vec<f32> = output.into_data().value.into_iter()
        .map(|v| v * metadata.target_std + metadata.target_mean)
        .collect();

    Ok(dates.into_iter().zip(predictions).collect())
}

fn preprocess_batch<B: Backend>(
    records: &[Record],
    seq_length: usize,
    device: &B::Device,
    feature_means: &[f32],
    feature_stds: &[f32],
) -> Result<(Tensor<B, 3>, Vec<NaiveDate>), AppError> {
    if records.len() < seq_length + 1 {
        return Err(AppError::InvalidData("Insufficient data for sequence".into()));
    }

    let mut sequences = Vec::with_capacity(records.len() - seq_length);
    let mut dates = Vec::with_capacity(records.len() - seq_length);

    for i in 0..records.len() - seq_length {
        let seq: Result<Vec<Vec<f32>>, AppError> = records[i..i + seq_length]
            .iter()
            .map(|record| {
                let row = parse_row(record)?;
                Ok(row.into_iter().take(5).enumerate()
                    .map(|(j, x)| (x - feature_means[j]) / feature_stds[j])
                    .collect())
            })
            .collect();
        let seq = seq?;
        sequences.push(seq);
        let date = NaiveDate::parse_from_str(&records[i + seq_length].data, "%Y-%m-%d")
            .map_err(|_| AppError::InvalidDate(records[i + seq_length].data.clone()))?;
        dates.push(date);
    }

    if sequences.is_empty() {
        return Err(AppError::InvalidData("No sequences generated".into()));
    }

    let flattened: Vec<f32> = sequences.into_iter().flatten().flatten().collect();
    let x = Tensor::<B, 1>::from_floats(flattened.as_slice(), device)
        .reshape([dates.len(), seq_length, feature_means.len()]);

    Ok((x, dates))
}