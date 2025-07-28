// file : src/conexao/read_file.rs




use burn::tensor::{backend::Backend, Tensor};
use chrono::NaiveDate;
use csv::ReaderBuilder;
use std::fs::File;
use crate::utils::{AppError, Record, parse_row};

pub fn read_raw_data(file_path: &str) -> Result<Vec<Record>, AppError> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }
    Ok(records)
}

pub fn ler_csv<B: Backend>(
    file_path: &str,
    seq_length: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2>, Vec<NaiveDate>, f32, f32, [f32; 5], [f32; 5]), AppError> {
    let records = read_raw_data(file_path)?;
    if records.len() < seq_length + 1 {
        return Err(AppError::InvalidData("Insufficient data for sequence".into()));
    }

    let parsed_data: Vec<Vec<f32>> = records.iter()
        .map(parse_row)
        .collect::<Result<Vec<_>, _>>()?;

    let feature_means: [f32; 5] = {
        let mut sums = [0.0; 5];
        for row in &parsed_data {
            for (i, &value) in row.iter().take(5).enumerate() {
                sums[i] += value;
            }
        }
        sums.map(|sum| sum / parsed_data.len() as f32)
    };

    let feature_stds: [f32; 5] = {
        let mut variances = [0.0; 5];
        for row in &parsed_data {
            for (i, &value) in row.iter().take(5).enumerate() {
                variances[i] += (value - feature_means[i]).powi(2);
            }
        }
        variances.map(|var| (var / parsed_data.len() as f32).sqrt().max(1e-8))
    };

    let mut sequences = Vec::with_capacity(records.len() - seq_length);
    let mut targets = Vec::with_capacity(records.len() - seq_length);
    let mut dates = Vec::with_capacity(records.len() - seq_length);

    for i in 0..records.len() - seq_length {
        let seq: Vec<Vec<f32>> = parsed_data[i..i + seq_length]
            .iter()
            .map(|row| {
                row.iter().take(5).enumerate()
                    .map(|(j, &x)| (x - feature_means[j]) / feature_stds[j])
                    .collect()
            })
            .collect();
        sequences.push(seq);

        let target = parsed_data[i + seq_length][5]; // Closing price
        targets.push(target);

        let date = NaiveDate::parse_from_str(&records[i + seq_length].data, "%Y-%m-%d")
            .map_err(|_| AppError::InvalidDate(records[i + seq_length].data.clone()))?;
        dates.push(date);
    }

    let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let target_std = (targets.iter().map(|&x| (x - target_mean).powi(2)).sum::<f32>() / targets.len() as f32).sqrt();

    let targets_normalized: Vec<f32> = targets.into_iter()
        .map(|x| (x - target_mean) / target_std.max(1e-8))
        .collect();

    let flattened_sequences: Vec<f32> = sequences.into_iter().flatten().flatten().collect();
    let x = Tensor::<B, 1>::from_floats(flattened_sequences.as_slice(), device)
        .reshape([dates.len(), seq_length, 5]);

    let y = Tensor::<B, 1>::from_floats(targets_normalized.as_slice(), device)
        .reshape([dates.len(), 1]);

    Ok((x, y, dates, target_mean, target_std, feature_means, feature_stds))
}