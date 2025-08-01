// projeto : lstmfilepredict
// file : src/main.rs


// projeto : lstmfilepredict
// file : src/main.rs - Versão Corrigida

use clap::Parser;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls};
use postgres_types::Json;
use rusqlite::{Connection, params};
use chrono::{NaiveDate, Duration};
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version = "2.1.0", about = "LSTM Stock Price Prediction - Fixed Version", long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long, default_value = "../dados")]
    data_dir: String,
    #[arg(long, default_value_t = 5)]
    num_predictions: usize,
    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
    #[arg(long, help = "Show detailed analysis")]
    verbose: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockData {
    date: String,
    closing: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelWeights {
    asset: String,
    source: String,
    // LSTM weights - corrigido para separar W e U
    w_input: Array2<f32>,
    u_input: Array2<f32>,
    b_input: Array1<f32>,
    
    w_forget: Array2<f32>,
    u_forget: Array2<f32>,
    b_forget: Array1<f32>,
    
    w_output: Array2<f32>,
    u_output: Array2<f32>,
    b_output: Array1<f32>,
    
    w_cell: Array2<f32>,
    u_cell: Array2<f32>,
    b_cell: Array1<f32>,
    
    // Output layer
    w_final: Array1<f32>,
    b_final: f32,
    
    // Normalization - nomes corrigidos
    data_mean: f32,
    data_std: f32,
    seq_length: usize,
    hidden_size: usize,
    timestamp: String,
}

struct LSTMCell {
    hidden_size: usize,
    w_input: Array2<f32>,
    u_input: Array2<f32>,
    b_input: Array1<f32>,
    
    w_forget: Array2<f32>,
    u_forget: Array2<f32>,
    b_forget: Array1<f32>,
    
    w_output: Array2<f32>,
    u_output: Array2<f32>,
    b_output: Array1<f32>,
    
    w_cell: Array2<f32>,
    u_cell: Array2<f32>,
    b_cell: Array1<f32>,
    
    w_final: Array1<f32>,
    b_final: f32,
}

impl LSTMCell {
    fn from_weights(weights: &ModelWeights) -> Self {
        LSTMCell {
            hidden_size: weights.hidden_size,
            w_input: weights.w_input.clone(),
            u_input: weights.u_input.clone(),
            b_input: weights.b_input.clone(),
            
            w_forget: weights.w_forget.clone(),
            u_forget: weights.u_forget.clone(),
            b_forget: weights.b_forget.clone(),
            
            w_output: weights.w_output.clone(),
            u_output: weights.u_output.clone(),
            b_output: weights.b_output.clone(),
            
            w_cell: weights.w_cell.clone(),
            u_cell: weights.u_cell.clone(),
            b_cell: weights.b_cell.clone(),
            
            w_final: weights.w_final.clone(),
            b_final: weights.b_final,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        if x > 500.0 { return 1.0; }
        if x < -500.0 { return 0.0; }
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f32) -> f32 {
        if x > 20.0 { return 1.0; }
        if x < -20.0 { return -1.0; }
        x.tanh()
    }

    fn forward_sequence(&self, sequence: &Array2<f32>) -> f32 {
        let seq_len = sequence.shape()[0];
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);

        for t in 0..seq_len {
            let x = sequence.row(t);
            
            // Compute gates usando a arquitetura correta
            let i_t = (&self.w_input.dot(&x) + &self.u_input.dot(&h) + &self.b_input)
                .mapv(Self::sigmoid);
            let f_t = (&self.w_forget.dot(&x) + &self.u_forget.dot(&h) + &self.b_forget)
                .mapv(Self::sigmoid);
            let o_t = (&self.w_output.dot(&x) + &self.u_output.dot(&h) + &self.b_output)
                .mapv(Self::sigmoid);
            let g_t = (&self.w_cell.dot(&x) + &self.u_cell.dot(&h) + &self.b_cell)
                .mapv(Self::tanh);

            // Update states
            c = &f_t * &c + &i_t * &g_t;
            h = &o_t * &c.mapv(Self::tanh);
        }

        self.w_final.dot(&h) + self.b_final
    }
    
    // Método para single step prediction (para previsões iterativas)
    fn forward_single(&self, input: f32, h_prev: &Array1<f32>, c_prev: &Array1<f32>) 
        -> (f32, Array1<f32>, Array1<f32>) {
        
        let x = Array1::from_vec(vec![input]);
        
        let i_t = (&self.w_input.dot(&x) + &self.u_input.dot(h_prev) + &self.b_input)
            .mapv(Self::sigmoid);
        let f_t = (&self.w_forget.dot(&x) + &self.u_forget.dot(h_prev) + &self.b_forget)
            .mapv(Self::sigmoid);
        let o_t = (&self.w_output.dot(&x) + &self.u_output.dot(h_prev) + &self.b_output)
            .mapv(Self::sigmoid);
        let g_t = (&self.w_cell.dot(&x) + &self.u_cell.dot(h_prev) + &self.b_cell)
            .mapv(Self::tanh);

        let c_new = &f_t * c_prev + &i_t * &g_t;
        let h_new = &o_t * &c_new.mapv(Self::tanh);
        
        let output = self.w_final.dot(&h_new) + self.b_final;
        
        (output, h_new, c_new)
    }
}

fn load_data_from_sqlite(data_dir: &str, source: &str, asset: &str) -> Result<Vec<StockData>, Box<dyn Error>> {
    let db_path = format!("{}/{}.db", data_dir, source);
    let conn = Connection::open(&db_path)?;

    let mut stmt = conn.prepare(
        "SELECT date, closing FROM stock_records WHERE asset = ?1 ORDER BY date ASC",
    )?;

    let records: Result<Vec<StockData>, _> = stmt.query_map(params![asset], |row| {
        Ok(StockData {
            date: row.get(0)?,
            closing: row.get(1)?,
        })
    })?.collect();

    Ok(records?)
}

fn load_model_from_postgres(pg_client: &mut Client, asset: &str, source: &str) -> Result<ModelWeights, Box<dyn Error>> {
    let query = "SELECT weights_json FROM lstm_weights WHERE asset = $1 AND source = $2";
    let rows = pg_client.query(query, &[&asset, &source])?;

    if rows.is_empty() {
        return Err(format!("No model found for asset {} and source {}", asset, source).into());
    }

    let weights_json: Json<ModelWeights> = rows[0].get(0);
    Ok(weights_json.0)
}

fn normalize_sequence(prices: &[f32], mean: f32, std: f32) -> Vec<f32> {
    prices.iter().map(|&x| (x - mean) / std).collect()
}

fn denormalize_price(normalized: f32, mean: f32, std: f32) -> f32 {
    normalized * std + mean
}

fn calculate_prediction_confidence(predictions: &[f32], historical_volatility: f32) -> Vec<f32> {
    let pred_mean = predictions.iter().sum::<f32>() / predictions.len() as f32;
    let pred_std = (predictions.iter().map(|x| (x - pred_mean).powi(2)).sum::<f32>() 
        / predictions.len() as f32).sqrt();
    
    predictions.iter().map(|&pred| {
        let z_score = ((pred - pred_mean) / pred_std.max(0.001)).abs();
        (1.0 - z_score.min(3.0) / 3.0) * (1.0 - historical_volatility.min(0.5))
    }).collect()
}

fn predict_future_prices(
    lstm: &LSTMCell, 
    initial_sequence: &[f32], 
    num_predictions: usize,
    mean: f32, 
    std: f32,
    verbose: bool
) -> Vec<f32> {
    let mut predictions = Vec::new();
    let seq_length = initial_sequence.len();
    
    // Preparar sequência inicial como Array2
    let mut current_sequence = Array2::zeros((seq_length, 1));
    for i in 0..seq_length {
        current_sequence[[i, 0]] = initial_sequence[i];
    }
    
    if verbose {
        println!("\n=== Prediction Process ===");
        println!("Initial sequence (normalized): {:?}", 
                initial_sequence.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    }

    // Fazer previsões iterativas
    for step in 0..num_predictions {
        let normalized_pred = lstm.forward_sequence(&current_sequence);
        let denormalized_pred = denormalize_price(normalized_pred, mean, std);
        predictions.push(denormalized_pred);
        
        if verbose {
            println!("Step {}: Normalized = {:.6}, Denormalized = {:.2}", 
                    step + 1, normalized_pred, denormalized_pred);
        }
        
        // Atualizar sequência: remover primeiro elemento e adicionar predição
        let mut new_sequence = Array2::zeros((seq_length, 1));
        for i in 0..(seq_length - 1) {
            new_sequence[[i, 0]] = current_sequence[[i + 1, 0]];
        }
        new_sequence[[seq_length - 1, 0]] = normalized_pred;
        current_sequence = new_sequence;
    }

    predictions
}

fn analyze_historical_performance(
    lstm: &LSTMCell,
    data: &[StockData],
    weights: &ModelWeights,
    verbose: bool
) -> (f32, f32) {
    if data.len() < weights.seq_length + 10 {
        return (0.0, 0.0);
    }

    let prices: Vec<f32> = data.iter().map(|d| d.closing).collect();
    let normalized_prices = normalize_sequence(&prices, weights.data_mean, weights.data_std);
    
    let mut errors = Vec::new();
    let mut correct_directions = 0;
    let mut total_predictions = 0;
    
    // Test on last 20 samples
    let test_start = data.len().saturating_sub(30);
    let test_end = data.len().saturating_sub(weights.seq_length);
    
    for i in test_start..test_end {
        if i + weights.seq_length < normalized_prices.len() {
            let sequence: Vec<f32> = normalized_prices[i..i + weights.seq_length].to_vec();
            let mut seq_array = Array2::zeros((weights.seq_length, 1));
            for j in 0..weights.seq_length {
                seq_array[[j, 0]] = sequence[j];
            }
            
            let predicted_norm = lstm.forward_sequence(&seq_array);
            let predicted_price = denormalize_price(predicted_norm, weights.data_mean, weights.data_std);
            let actual_price = prices[i + weights.seq_length];
            
            let error = (predicted_price - actual_price).abs();
            errors.push(error);
            
            // Check direction
            let last_price = prices[i + weights.seq_length - 1];
            let predicted_direction = predicted_price > last_price;
            let actual_direction = actual_price > last_price;
            
            if predicted_direction == actual_direction {
                correct_directions += 1;
            }
            total_predictions += 1;
            
            if verbose {
                println!("Historical test {}: Last={:.2}, Pred={:.2}, Actual={:.2}, Error={:.2}", 
                        i, last_price, predicted_price, actual_price, error);
            }
        }
    }
    
    let avg_error = if errors.is_empty() { 0.0 } else { 
        errors.iter().sum::<f32>() / errors.len() as f32 
    };
    let direction_accuracy = if total_predictions == 0 { 0.0 } else {
        correct_directions as f32 / total_predictions as f32
    };
    
    (avg_error, direction_accuracy)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== LSTM Stock Price Prediction (Fixed Version) ===");
    println!("Asset: {}, Source: {}", args.asset, args.source);
    println!("Number of Predictions: {}", args.num_predictions);

    // Connect to PostgreSQL and load model
    println!("\n📡 Loading model from PostgreSQL...");
    let mut pg_client = Client::connect(&args.pg_conn, NoTls)?;
    let weights = load_model_from_postgres(&mut pg_client, &args.asset, &args.source)?;
    let lstm = LSTMCell::from_weights(&weights);

    println!("✅ Model loaded successfully!");
    println!("   🧠 Hidden Size: {}", weights.hidden_size);
    println!("   📏 Sequence Length: {}", weights.seq_length);
    println!("   📊 Normalization: Mean={:.4}, Std={:.4}", weights.data_mean, weights.data_std);
    println!("   📅 Model Timestamp: {}", weights.timestamp);

    // Load historical data
    println!("\n📈 Loading historical data...");
    let data = load_data_from_sqlite(&args.data_dir, &args.source, &args.asset)?;
    
    if data.len() < weights.seq_length {
        return Err(format!("Insufficient historical data: need {}, got {}", 
                          weights.seq_length, data.len()).into());
    }

    println!("✅ Loaded {} historical records", data.len());
    
    // Show recent data
    println!("\n📋 Recent historical data:");
    for record in data.iter().rev().take(5).rev() {
        println!("   {}: {:.2} BRL", record.date, record.closing);
    }

    // Analyze historical performance
    println!("\n🔍 Analyzing model performance on historical data...");
    let (avg_error, direction_accuracy) = analyze_historical_performance(&lstm, &data, &weights, args.verbose);
    println!("   📊 Average Prediction Error: {:.2} BRL", avg_error);
    println!("   🎯 Directional Accuracy: {:.1}%", direction_accuracy * 100.0);

    // Prepare input sequence (most recent prices)
    let recent_prices: Vec<f32> = data.iter()
        .rev()
        .take(weights.seq_length)
        .rev()
        .map(|r| r.closing)
        .collect();
    
    let normalized_sequence = normalize_sequence(&recent_prices, weights.data_mean, weights.data_std);
    
    if args.verbose {
        println!("\n📊 Input sequence preparation:");
        for (i, (price, norm)) in recent_prices.iter().zip(normalized_sequence.iter()).enumerate() {
            println!("   Position {}: Price={:.2} → Normalized={:.4}", i, price, norm);
        }
    }

    // Make predictions
    println!("\n🔮 Generating future price predictions...");
    let predictions = predict_future_prices(
        &lstm, 
        &normalized_sequence, 
        args.num_predictions,
        weights.data_mean, 
        weights.data_std,
        args.verbose
    );

    // Calculate confidence scores
    let last_prices: Vec<f32> = recent_prices.iter().cloned().collect();
    let historical_volatility = {
        let returns: Vec<f32> = last_prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        if returns.is_empty() { 0.1 } else {
            let mean_return = returns.iter().sum::<f32>() / returns.len() as f32;
            (returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f32>() / returns.len() as f32).sqrt()
        }
    };
    let confidence_scores = calculate_prediction_confidence(&predictions, historical_volatility);

    // Generate future dates
    let last_date = &data.last().unwrap().date;
    let start_date = NaiveDate::parse_from_str(last_date, "%d.%m.%Y")
        .or_else(|_| NaiveDate::parse_from_str(last_date, "%Y-%m-%d"))
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2025, 7, 29).unwrap());

    // Display predictions
    println!("\n📈 Future Price Predictions:");
    println!("   Current Price: {:.2} BRL ({})", recent_prices.last().unwrap(), last_date);
    println!("   ┌─────────────┬─────────┬────────────┬────────────┐");
    println!("   │    Date     │  Price  │   Change   │ Confidence │");
    println!("   ├─────────────┼─────────┼────────────┼────────────┤");

    let current_price = *recent_prices.last().unwrap();
    for (i, (&price, &confidence)) in predictions.iter().zip(confidence_scores.iter()).enumerate() {
        let future_date = start_date + Duration::days(i as i64 + 1);
        let change = price - current_price;
        let change_pct = (change / current_price) * 100.0;
        
        let change_str = if change >= 0.0 {
            format!("+{:.2} ({:+.1}%)", change, change_pct)
        } else {
            format!("{:.2} ({:.1}%)", change, change_pct)
        };
        
        println!("   │ {} │ {:7.2} │ {:>10} │ {:>8.1}% │", 
                future_date.format("%Y-%m-%d"), 
                price, 
                change_str,
                confidence * 100.0);
    }
    println!("   └─────────────┴─────────┴────────────┴────────────┘");

    // Summary statistics
    let avg_prediction = predictions.iter().sum::<f32>() / predictions.len() as f32;
    let max_prediction = predictions.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_prediction = predictions.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let avg_confidence = confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32;

    println!("\n📊 Prediction Summary:");
    println!("   📈 Average Predicted Price: {:.2} BRL", avg_prediction);
    println!("   📊 Price Range: {:.2} - {:.2} BRL", min_prediction, max_prediction);
    println!("   📈 Expected Change: {:.2} BRL ({:+.1}%)", 
             avg_prediction - current_price, 
             ((avg_prediction - current_price) / current_price) * 100.0);
    println!("   🎯 Average Confidence: {:.1}%", avg_confidence * 100.0);
    println!("   📊 Historical Volatility: {:.1}%", historical_volatility * 100.0);

    // Warnings and disclaimers
    println!("\n⚠️  Important Notes:");
    println!("   • These predictions are based on historical patterns only");
    println!("   • Market conditions can change rapidly and unpredictably");
    println!("   • Use this information as one factor among many in decision making");
    println!("   • Consider consulting financial professionals for investment advice");
    
    if avg_confidence < 0.5 {
        println!("   ⚠️  Low confidence scores suggest high prediction uncertainty");
    }
    
    if direction_accuracy < 0.6 {
        println!("   ⚠️  Historical directional accuracy is below 60% - use predictions with caution");
    }

    Ok(())
}

// Exemplo de uso:
// cargo run -- --asset WEGE3 --source investing --num-predictions 5 --verbose
// cargo run -- --asset WEGE3 --source investing --num-predictions 10


// cd lstmfilepredict
// cargo run -- --asset WEGE3 --source investing --seq-length 20 --num-predictions 20

// cargo run -- predict --asset WEGE3 --source investing --num-predictions 20 
// cargo run -- --asset WEGE3 --source investing

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
