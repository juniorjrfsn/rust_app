// projeto: lstmrnntrain
// file: src/neural/metrics.rs
// Training metrics definitions and calculations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub asset: String,
    pub model_type: String,
    pub source: String,
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub directional_accuracy: f64,
    pub r_squared: f64,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct MetricsTracker {
    pub history: Vec<TrainingMetrics>,
    pub best_val_loss: f64,
    pub best_epoch: usize,
    pub patience_counter: usize,
    pub no_improvement_count: usize,
}

impl MetricsTracker {
    pub fn new() -> Self {
        MetricsTracker {
            history: Vec::new(),
            best_val_loss: f64::INFINITY,
            best_epoch: 0,
            patience_counter: 0,
            no_improvement_count: 0,
        }
    }

    pub fn add_metrics(&mut self, metrics: TrainingMetrics, patience: usize) -> bool {
        let is_best = metrics.val_loss < self.best_val_loss;
        
        if is_best {
            self.best_val_loss = metrics.val_loss;
            self.best_epoch = metrics.epoch;
            self.patience_counter = 0;
            self.no_improvement_count = 0;
            println!("🎯 [Metrics] New best validation loss: {:.6} at epoch {}", 
                     self.best_val_loss, self.best_epoch);
        } else {
            self.patience_counter += 1;
            self.no_improvement_count += 1;
        }

        self.history.push(metrics);

        // Return true if should stop training (early stopping)
        self.patience_counter >= patience
    }

    pub fn get_best_metrics(&self) -> Option<&TrainingMetrics> {
        self.history.iter()
            .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap())
    }

    pub fn get_latest_metrics(&self) -> Option<&TrainingMetrics> {
        self.history.last()
    }

    pub fn print_summary(&self) {
        if let Some(best) = self.get_best_metrics() {
            println!("📈 [Metrics] Training Summary:");
            println!("   ├── Best Epoch: {}", best.epoch);
            println!("   ├── Best Val Loss: {:.6}", best.val_loss);
            println!("   ├── RMSE: {:.6}", best.rmse);
            println!("   ├── MAE: {:.6}", best.mae);
            println!("   ├── MAPE: {:.2}%", best.mape);
            println!("   ├── Direction Acc: {:.2}%", best.directional_accuracy * 100.0);
            println!("   └── R²: {:.6}", best.r_squared);
        }
    }

    pub fn save_to_csv(&self, file_path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(file_path)?;
        
        // Write header
        writeln!(file, "asset,model_type,epoch,train_loss,val_loss,rmse,mae,mape,directional_accuracy,r_squared,timestamp")?;
        
        // Write data
        for metrics in &self.history {
            writeln!(
                file,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.2},{:.4},{:.6},{}",
                metrics.asset,
                metrics.model_type,
                metrics.epoch,
                metrics.train_loss,
                metrics.val_loss,
                metrics.rmse,
                metrics.mae,
                metrics.mape,
                metrics.directional_accuracy,
                metrics.r_squared,
                metrics.timestamp
            )?;
        }
        
        println!("📊 [Metrics] Training history saved to: {}", file_path);
        Ok(())
    }
}

/// Calculate comprehensive metrics for model evaluation
pub fn calculate_regression_metrics(predictions: &[f64], targets: &[f64]) -> RegressionMetrics {
    assert_eq!(predictions.len(), targets.len(), "Predictions and targets must have same length");
    
    let n = predictions.len() as f64;
    
    // Mean Squared Error and RMSE
    let mse = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n;
    let rmse = mse.sqrt();

    // Mean Absolute Error
    let mae = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / n;

    // Mean Absolute Percentage Error
    let mape = predictions.iter().zip(targets.iter())
        .filter(|(_, t)| **t != 0.0)
        .map(|(p, t)| ((p - t) / t).abs())
        .sum::<f64>() / n * 100.0;

    // Directional Accuracy
    let mut correct_direction = 0;
    for i in 1..predictions.len() {
        let pred_change = predictions[i] - predictions[i-1];
        let actual_change = targets[i] - targets[i-1];
        if pred_change.signum() == actual_change.signum() {
            correct_direction += 1;
        }
    }
    let directional_accuracy = if predictions.len() > 1 {
        correct_direction as f64 / (predictions.len() - 1) as f64
    } else {
        0.0
    };

    // R-squared
    let target_mean = targets.iter().sum::<f64>() / n;
    let ss_res = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum::<f64>();
    let ss_tot = targets.iter()
        .map(|t| (t - target_mean).powi(2))
        .sum::<f64>();
    let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

    // Additional metrics
    let max_error = predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .fold(0.0, |acc, x| acc.max(x));

    let median_ae = {
        let mut abs_errors: Vec<f64> = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();
        abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = abs_errors.len() / 2;
        if abs_errors.len() % 2 == 0 {
            (abs_errors[mid - 1] + abs_errors[mid]) / 2.0
        } else {
            abs_errors[mid]
        }
    };

    RegressionMetrics {
        mse,
        rmse,
        mae,
        mape,
        directional_accuracy,
        r_squared,
        max_error,
        median_ae,
        n_samples: predictions.len(),
    }
}

#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub directional_accuracy: f64,
    pub r_squared: f64,
    pub max_error: f64,
    pub median_ae: f64,
    pub n_samples: usize,
}

impl RegressionMetrics {
    pub fn print(&self, prefix: &str) {
        println!("📊 [{}] Regression Metrics:", prefix);
        println!("   ├── Samples: {}", self.n_samples);
        println!("   ├── RMSE: {:.6}", self.rmse);
        println!("   ├── MAE: {:.6}", self.mae);
        println!("   ├── MAPE: {:.2}%", self.mape);
        println!("   ├── Max Error: {:.6}", self.max_error);
        println!("   ├── Median AE: {:.6}", self.median_ae);
        println!("   ├── Direction Acc: {:.2}%", self.directional_accuracy * 100.0);
        println!("   └── R²: {:.6}", self.r_squared);
    }

    pub fn is_better_than(&self, other: &RegressionMetrics, primary_metric: &str) -> bool {
        match primary_metric.to_lowercase().as_str() {
            "rmse" => self.rmse < other.rmse,
            "mae" => self.mae < other.mae,
            "mape" => self.mape < other.mape,
            "r2" | "r_squared" => self.r_squared > other.r_squared,
            "directional_accuracy" => self.directional_accuracy > other.directional_accuracy,
            _ => self.rmse < other.rmse, // Default to RMSE
        }
    }
}

/// Performance comparison utilities
#[derive(Debug)]
pub struct ModelComparison {
    pub models: HashMap<String, Vec<RegressionMetrics>>,
}

impl ModelComparison {
    pub fn new() -> Self {
        ModelComparison {
            models: HashMap::new(),
        }
    }

    pub fn add_model_metrics(&mut self, model_name: String, metrics: RegressionMetrics) {
        self.models.entry(model_name).or_insert_with(Vec::new).push(metrics);
    }

    pub fn compare_models(&self, metric: &str) {
        println!("🏆 [Comparison] Model Performance Comparison ({})", metric);
        println!("   {:<15} {:<12} {:<12} {:<12}", "Model", "Best", "Average", "Std Dev");
        println!("   {}", "─".repeat(55));

        let mut model_stats: Vec<(String, f64, f64, f64)> = Vec::new();

        for (model_name, metrics_list) in &self.models {
            if metrics_list.is_empty() { continue; }

            let values: Vec<f64> = metrics_list.iter().map(|m| match metric.to_lowercase().as_str() {
                "rmse" => m.rmse,
                "mae" => m.mae,
                "mape" => m.mape,
                "r2" | "r_squared" => m.r_squared,
                "directional_accuracy" => m.directional_accuracy,
                _ => m.rmse,
            }).collect();

            let best = if metric.to_lowercase().as_str() == "r2" || 
                         metric.to_lowercase().as_str() == "r_squared" ||
                         metric.to_lowercase().as_str() == "directional_accuracy" {
                values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            } else {
                values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            };

            let average = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (x - average).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            model_stats.push((model_name.clone(), best, average, std_dev));
        }

        // Sort by best performance
        model_stats.sort_by(|a, b| {
            if metric.to_lowercase().as_str() == "r2" || 
               metric.to_lowercase().as_str() == "r_squared" ||
               metric.to_lowercase().as_str() == "directional_accuracy" {
                b.1.partial_cmp(&a.1).unwrap() // Higher is better
            } else {
                a.1.partial_cmp(&b.1).unwrap() // Lower is better
            }
        });

        for (i, (model_name, best, average, std_dev)) in model_stats.iter().enumerate() {
            let prefix = if i == 0 { "🥇" } else if i == 1 { "🥈" } else if i == 2 { "🥉" } else { "  " };
            println!("   {}{:<13} {:<12.6} {:<12.6} {:<12.6}", 
                     prefix, model_name, best, average, std_dev);
        }
    }

    pub fn export_comparison(&self, file_path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(file_path)?;
        
        writeln!(file, "model_name,rmse,mae,mape,directional_accuracy,r_squared,max_error,median_ae,n_samples")?;
        
        for (model_name, metrics_list) in &self.models {
            for metrics in metrics_list {
                writeln!(
                    file,
                    "{},{:.6},{:.6},{:.2},{:.4},{:.6},{:.6},{:.6},{}",
                    model_name,
                    metrics.rmse,
                    metrics.mae,
                    metrics.mape,
                    metrics.directional_accuracy,
                    metrics.r_squared,
                    metrics.max_error,
                    metrics.median_ae,
                    metrics.n_samples
                )?;
            }
        }
        
        println!("📊 [Comparison] Model comparison exported to: {}", file_path);
        Ok(())
    }
}

/// Early stopping implementation
#[derive(Debug)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub mode: String, // "min" or "max"
    pub best_score: Option<f64>,
    pub wait: usize,
    pub stopped_epoch: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, mode: String) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            mode,
            best_score: None,
            wait: 0,
            stopped_epoch: 0,
        }
    }

    pub fn should_stop(&mut self, current_score: f64, epoch: usize) -> bool {
        match self.best_score {
            None => {
                self.best_score = Some(current_score);
                false
            },
            Some(best) => {
                let is_better = match self.mode.as_str() {
                    "min" => current_score < best - self.min_delta,
                    "max" => current_score > best + self.min_delta,
                    _ => current_score < best - self.min_delta, // Default to min
                };

                if is_better {
                    self.best_score = Some(current_score);
                    self.wait = 0;
                    false
                } else {
                    self.wait += 1;
                    if self.wait >= self.patience {
                        self.stopped_epoch = epoch;
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }

    pub fn get_best_score(&self) -> Option<f64> {
        self.best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_metrics() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.1, 1.9, 3.1, 3.8, 5.2];
        
        let metrics = calculate_regression_metrics(&predictions, &targets);
        
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.mape >= 0.0);
        assert!(metrics.directional_accuracy >= 0.0 && metrics.directional_accuracy <= 1.0);
        assert!(metrics.r_squared <= 1.0);
        assert_eq!(metrics.n_samples, 5);
    }

    #[test]
    fn test_early_stopping() {
        let mut early_stopping = EarlyStopping::new(3, 0.001, "min".to_string());
        
        // Should not stop initially
        assert!(!early_stopping.should_stop(1.0, 1));
        assert!(!early_stopping.should_stop(0.9, 2)); // Improvement
        assert!(!early_stopping.should_stop(0.91, 3)); // No significant improvement
        assert!(!early_stopping.should_stop(0.92, 4)); // No significant improvement
        assert!(early_stopping.should_stop(0.93, 5)); // Should stop after patience
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        
        let metrics = TrainingMetrics {
            asset: "TEST".to_string(),
            model_type: "LSTM".to_string(),
            source: "test".to_string(),
            epoch: 1,
            train_loss: 0.5,
            val_loss: 0.6,
            rmse: 0.1,
            mae: 0.08,
            mape: 5.0,
            directional_accuracy: 0.65,
            r_squared: 0.8,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        
        let should_stop = tracker.add_metrics(metrics, 10);
        assert!(!should_stop);
        assert_eq!(tracker.history.len(), 1);
        assert_eq!(tracker.best_val_loss, 0.6);
        assert_eq!(tracker.best_epoch, 1);
    }

    #[test]
    fn test_model_comparison() {
        let mut comparison = ModelComparison::new();
        
        let metrics1 = RegressionMetrics {
            mse: 0.01,
            rmse: 0.1,
            mae: 0.08,
            mape: 5.0,
            directional_accuracy: 0.65,
            r_squared: 0.8,
            max_error: 0.2,
            median_ae: 0.07,
            n_samples: 100,
        };
        
        let metrics2 = RegressionMetrics {
            mse: 0.02,
            rmse: 0.14,
            mae: 0.12,
            mape: 7.0,
            directional_accuracy: 0.60,
            r_squared: 0.75,
            max_error: 0.25,
            median_ae: 0.10,
            n_samples: 100,
        };
        
        comparison.add_model_metrics("LSTM".to_string(), metrics1.clone());
        comparison.add_model_metrics("RNN".to_string(), metrics2.clone());
        
        assert!(metrics1.is_better_than(&metrics2, "rmse"));
        assert!(!metrics2.is_better_than(&metrics1, "r_squared"));
    }
}