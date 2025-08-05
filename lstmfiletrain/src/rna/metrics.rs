// projeto: lstmfiletrain
// file: src/rna/metrics.rs

pub fn calculate_directional_accuracy(predictions: &[f32], targets: &[f32], sequences: &[Vec<f32>]) -> f32 {
    if predictions.len() != targets.len() || predictions.is_empty() || sequences.len() != targets.len() {
        return 0.0;
    }
    let mut correct = 0;
    for i in 0..predictions.len() {
        if let Some(&last_closing) = sequences[i].first() {
            let pred_direction = predictions[i] > last_closing;
            let actual_direction = targets[i] > last_closing;
            if pred_direction == actual_direction {
                correct += 1;
            }
        }
    }
    correct as f32 / predictions.len() as f32
}

pub fn calculate_mape(predictions: &[f32], targets: &[f32]) -> f32 {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return 0.0;
    }
    let mut total_percentage_error = 0.0;
    let mut count = 0;
    for (&pred, &actual) in predictions.iter().zip(targets.iter()) {
        if actual.abs() > 1e-8 {
            total_percentage_error += ((pred - actual) / actual).abs();
            count += 1;
        }
    }
    if count > 0 {
        (total_percentage_error / count as f32) * 100.0
    } else {
        0.0
    }
}

pub fn calculate_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return f32::NAN;
    }
    let mse: f32 = predictions.iter()
        .zip(targets.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f32>() / predictions.len() as f32;
    mse.sqrt()
}