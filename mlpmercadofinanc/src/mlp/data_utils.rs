// file : src/mlp/data_utils.rs


use burn::tensor::{backend::Backend, Tensor};
use chrono::NaiveDate;
use crate::utils::{AppError, Record, parse_row};

pub fn preprocess_lstm_single<B: Backend>(
    records: &[Record],
    seq_length: usize,
    device: &B::Device,
    feature_means: &[f32; 5],
    feature_stds: &[f32; 5],
) -> Result<(Tensor<B, 3>, NaiveDate), AppError> {
    if records.len() < seq_length {
        return Err(AppError::InvalidData("Insufficient data for sequence".into()));
    }

    let seq_start = records.len() - seq_length;
    let seq: Result<Vec<Vec<f32>>, AppError> = records[seq_start..]
        .iter()
        .map(|record| {
            let row = parse_row(record)?;
            Ok(row.into_iter().take(5).enumerate()
                .map(|(i, x)| (x - feature_means[i]) / feature_stds[i])
                .collect())
        })
        .collect();
    let seq = seq?;

    let flattened_seq: Vec<f32> = seq.into_iter().flatten().collect();
    let x = Tensor::<B, 1>::from_floats(flattened_seq.as_slice(), device)
        .reshape([1, seq_length, 5]);

    let date = NaiveDate::parse_from_str(&records.last().unwrap().data, "%Y-%m-%d")
        .map_err(|_| AppError::InvalidDate(records.last().unwrap().data.clone()))?;

    Ok((x, date))
}