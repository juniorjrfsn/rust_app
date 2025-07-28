use clap::Parser;
use std::fs;
use serde::{Deserialize, Serialize};
use serde_json;
use toml;
use ndarray::{Array1, Array2, s};
use log::{info, error};
use env_logger;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long, default_value_t = 10)]
    seq_length: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockRecord {
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
}

#[derive(Debug, Deserialize)]
struct StockData {
    records: Vec<StockRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Vec<Vec<f32>>,
    weight_hh: Vec<Vec<f32>>,
    bias: Vec<f32>,
    weight_out: Vec<f32>,
    bias_out: f32,
}

impl LSTMCell {
    fn forward(
        &self,
        x: &Array1<f32>,
        h_prev: &Array1<f32>,
        c_prev: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, f32) {
        let gates = {
            let ih = Array2::from_shape_vec(
                (4 * self.hidden_size, self.input_size),
                self.weight_ih.iter().flatten().cloned().collect(),
            )
            .unwrap();
            let hh = Array2::from_shape_vec(
                (4 * self.hidden_size, self.hidden_size),
                self.weight_hh.iter().flatten().cloned().collect(),
            )
            .unwrap();
            let bias = Array1::from_vec(self.bias.clone());
            &ih.dot(x) + &hh.dot(h_prev) + &bias
        };
        let i = sigmoid(&gates.slice(s![..self.hidden_size]).to_owned());
        let f = sigmoid(&gates.slice(s![self.hidden_size..2 * self.hidden_size]).to_owned());
        let o = sigmoid(&gates.slice(s![2 * self.hidden_size..3 * self.hidden_size]).to_owned());
        let g = tanh(&gates.slice(s![3 * self.hidden_size..]).to_owned());

        let c = &f * c_prev + &i * &g;
        let h = &o * tanh(&c);

        let output = h.dot(&Array1::from_vec(self.weight_out.clone())) + self.bias_out;
        (h.to_owned(), c.to_owned(), output)
    }

    fn predict(&self, sequence: &[f32], mean: f32, std: f32) -> f32 {
        let mut h_t = Array1::zeros(self.hidden_size);
        let mut c_t = Array1::zeros(self.hidden_size);
        let mut output = 0.0;
        for &val in sequence {
            let x = Array1::from_vec(vec![val]);
            let (h_next, c_next, pred) = self.forward(&x, &h_t, &c_t);
            h_t = h_next;
            c_t = c_next;
            output = pred;
        }
        output * std + mean
    }
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let data_path = format!("../dados/{}_{}_output.toml", args.asset, args.source);
    let model_path = format!("../dados/{}_{}_lstm_model.json", args.asset, args.source);

    info!("Loading model from {}", model_path);
    let model_json = fs::read_to_string(&model_path)?;
    let lstm: LSTMCell = serde_json::from_str(&model_json)?;

    info!("Loading data from {}", data_path);
    let toml_data = fs::read_to_string(&data_path)?;
    let stock_data: StockData = toml::from_str(&toml_data)?;
    let records = stock_data.records;

    let prices: Array1<f32> = Array1::from_vec(records.iter().map(|r| r.closing).collect());
    let mean = prices.mean().unwrap_or(0.0);
    let std = prices.std_axis(ndarray::Axis(0), 0.0).into_scalar();

    let seq_length = args.seq_length;
    let n_predictions = 20;
    if records.len() < seq_length + n_predictions {
        error!("Not enough records for {} predictions with seq_length {}", n_predictions, seq_length);
        return Ok(());
    }
    let mut predictions = Vec::new();
    let mut actual_prices = Vec::new();
    let mut dates = Vec::new();
    for i in (records.len() - n_predictions - seq_length)..(records.len() - seq_length) {
        let sequence: Vec<f32> = records.iter()
            .skip(i)
            .take(seq_length)
            .map(|r| (r.closing - mean) / std)
            .collect();
        let pred = lstm.predict(&sequence, mean, std);
        predictions.push(pred);
        actual_prices.push(records[i + seq_length].closing);
        dates.push(records[i + seq_length].date.clone());
    }

    let output_data = serde_json::json!({
        "dates": dates,
        "actual": actual_prices,
        "predicted": predictions
    });
    let output_path = format!("../dados/{}_{}_predictions.json", args.asset, args.source);
    fs::write(&output_path, serde_json::to_string_pretty(&output_data)?)?;
    info!("Predictions saved to: {}", output_path);

    for (i, (date, (actual, pred))) in dates.iter().zip(actual_prices.iter().zip(predictions.iter())).enumerate() {
        println!("Prediction {} ({}): Actual = {:.2} BRL, Predicted = {:.2} BRL", i + 1, date, actual, pred);
    }

    Ok(())
}

// cargo run -- --asset WEGE3 --source investing

// cd lstmfilepredict

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
