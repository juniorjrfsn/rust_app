// projeto: lstmfilepredict
// file: src/main.rs
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json; // Adicionado para deserialização
use ndarray::{Array1, Array2};
use postgres::{Client, NoTls, Row};
use chrono::{NaiveDate, Duration}; // Duration é necessário
use thiserror::Error;
use log::{info, warn, error};
use env_logger;

#[derive(Error, Debug)]
#[allow(dead_code)]
enum LSTMError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] postgres::Error),
    #[error("Model not found for asset '{0}'")]
    ModelNotFound(String),
    #[error("Insufficient data for asset '{asset}': required {required}, actual {actual}")]
    InsufficientData { asset: String, required: usize, actual: usize },
    #[error("Prediction error: {0}")]
    PredictionError(String),
    #[error("Data loading error: {0}")]
    DataLoadingError(String),
    #[error("Model deserialization error: {0}")]
    DeserializationError(String),
    #[error("Date parsing error: {0}")]
    DateParseError(String),
    #[error("Model forward pass error: {0}")]
    ForwardError(String),
}

#[derive(Parser, Debug)]
#[command(author, version = "3.0.2", about = "LSTM Stock Price Prediction - Detailed Version compatible with lstmfiletrain", long_about = None)]
struct Args {
    #[arg(long, help = "Asset to predict (e.g., SLCE3). If not provided, predicts for all assets.")]
    asset: Option<String>, // Tornado opcional

    #[arg(long, default_value = "../dados", help = "Data directory (for compatibility, not used)")]
    data_dir: String,

    #[arg(long, default_value_t = 5, help = "Number of future predictions")]
    num_predictions: usize,

    #[arg(long, default_value_t = false, help = "Enable verbose logging")]
    verbose: bool,

    #[arg(long, default_value = "postgres://postgres:postgres@localhost:5432/lstm_db")]
    pg_conn: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct StockData {
    date: String,
    closing: f32,
    opening: f32,
    // high, low, volume, variation podem ser adicionados se necessário
}

#[derive(Debug, Deserialize, Serialize, Clone)] // Clone adicionado
struct LSTMLayerWeights {
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
}

#[derive(Debug, Deserialize, Serialize, Clone)] // Clone adicionado
struct ModelWeights {
    asset: String,
    source: String,
    layers: Vec<LSTMLayerWeights>,
    w_final: Array1<f32>,
    b_final: f32,
    closing_mean: f32,
    closing_std: f32,
    opening_mean: f32,
    opening_std: f32,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
    timestamp: String,
}

// --- Funções auxiliares ---

// Função para carregar dados do PostgreSQL para um ativo específico
// CORRIGIDA: Adicionado parâmetro verbose
fn load_data_from_postgres(client: &mut Client, asset: &str, verbose: bool) -> Result<Vec<StockData>, LSTMError> {
    info!("  📥 Executing SQL query to load data for asset: {}", asset);
    let query = "SELECT date, closing, opening FROM stock_records WHERE asset = $1 ORDER BY date ASC";
    let rows = client.query(query, &[&asset])?;

    let mut data = Vec::with_capacity(rows.len());
    info!("  📊 Fetched {} rows from database", rows.len());
    for (i, row) in rows.iter().enumerate() {
        data.push(StockData {
            date: row.get(0),
            closing: row.get(1),
            opening: row.get(2),
        });
        // Log detalhado para as primeiras e últimas linhas
        // CORRIGIDA: Usar o parâmetro verbose
        if verbose && (i < 3 || i >= rows.len().saturating_sub(3)) {
             info!("    Row {}: Date={}, Close={}, Open={}", i+1, row.get::<_, String>(0), row.get::<_, f32>(1), row.get::<_, f32>(2));
        }
    }
    info!("  ✅ Successfully loaded {} data points for {}", data.len(), asset);
    Ok(data)
}

// Estrutura para representar uma célula LSTM durante a predição
#[derive(Clone)] // Derive Clone para MultiLayerLSTM
struct LSTMCell {
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
}

impl LSTMCell {
    fn sigmoid(x: f32) -> f32 {
        // Clamping para evitar overflow
        let x_clamped = x.max(-500.0).min(500.0);
        1.0 / (1.0 + (-x_clamped).exp())
    }

    fn tanh(x: f32) -> f32 {
        // Clamping para evitar overflow
        let x_clamped = x.max(-20.0).min(20.0);
        x_clamped.tanh()
    }

    fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        // Input Gate
        let i_t = (&self.w_input.dot(input) + &self.u_input.dot(h_prev) + &self.b_input).mapv(Self::sigmoid);
        // Forget Gate
        let f_t = (&self.w_forget.dot(input) + &self.u_forget.dot(h_prev) + &self.b_forget).mapv(Self::sigmoid);
        // Output Gate
        let o_t = (&self.w_output.dot(input) + &self.u_output.dot(h_prev) + &self.b_output).mapv(Self::sigmoid);
        // Cell State Candidate
        let c_hat_t = (&self.w_cell.dot(input) + &self.u_cell.dot(h_prev) + &self.b_cell).mapv(Self::tanh);
        // New Cell State
        let c_t = (&f_t * c_prev) + (&i_t * &c_hat_t);
        // New Hidden State
        let h_t = &o_t * c_t.mapv(Self::tanh);
        (h_t, c_t)
    }
}

// Estrutura para o modelo LSTM durante a predição
#[derive(Clone)] // Derive Clone
struct MultiLayerLSTM {
    layers: Vec<LSTMCell>,
    w_final: Array1<f32>,
    b_final: f32,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl MultiLayerLSTM {
    // Cria uma instância do modelo a partir dos pesos serializados
    fn from_weights(weights: &ModelWeights) -> Self { // Recebe uma referência
        info!("  🏗️ Building MultiLayerLSTM model from weights...");
        info!("    Layers: {}, Hidden Units: {}, Sequence Length: {}", weights.num_layers, weights.hidden_size, weights.seq_length);
        let layers = weights.layers.iter().map(|lw| {
            // Clonando os arrays para o modelo
            LSTMCell {
                w_input: lw.w_input.clone(),
                u_input: lw.u_input.clone(),
                b_input: lw.b_input.clone(),
                w_forget: lw.w_forget.clone(),
                u_forget: lw.u_forget.clone(),
                b_forget: lw.b_forget.clone(),
                w_output: lw.w_output.clone(),
                u_output: lw.u_output.clone(),
                b_output: lw.b_output.clone(),
                w_cell: lw.w_cell.clone(),
                u_cell: lw.u_cell.clone(),
                b_cell: lw.b_cell.clone(),
            }
        }).collect::<Vec<LSTMCell>>(); // Coletando explicitamente

        info!("  ✅ Model built successfully with {} layers", layers.len());
        Self {
            layers,
            w_final: weights.w_final.clone(),
            b_final: weights.b_final,
            seq_length: weights.seq_length,
            hidden_size: weights.hidden_size,
            num_layers: weights.num_layers,
        }
    }

    // Função forward para predição
    fn forward(&self, sequence: &[f32]) -> Result<f32, LSTMError> {
        if sequence.len() % 4 != 0 {
             return Err(LSTMError::ForwardError(format!("Invalid sequence length: {}. Expected a multiple of 4 (4 features per timestep).", sequence.len())));
        }
        let seq_timesteps = sequence.len() / 4;
        if seq_timesteps != self.seq_length {
             return Err(LSTMError::ForwardError(format!("Sequence length mismatch: got {} timesteps, expected {}.", seq_timesteps, self.seq_length)));
        }

        let mut h_states: Vec<Array1<f32>> = vec![Array1::zeros(self.hidden_size); self.num_layers];
        let mut c_states: Vec<Array1<f32>> = vec![Array1::zeros(self.hidden_size); self.num_layers];

        // Processa a sequência inteira passo a passo
        for i in 0..seq_timesteps {
            let start_idx = i * 4;
            let end_idx = start_idx + 4;
            let input_vec: Vec<f32> = sequence[start_idx..end_idx].to_vec();
            let mut layer_input = Array1::from_vec(input_vec);

            for (j, layer) in self.layers.iter().enumerate() {
                let (h_new, c_new) = layer.forward(&layer_input, &h_states[j], &c_states[j]);
                h_states[j] = h_new.clone();
                c_states[j] = c_new;
                layer_input = h_new; // Saída da camada anterior é entrada da próxima
            }
        }
        // Saída final (camada densa)
        let output = self.w_final.dot(&h_states[self.num_layers - 1]) + self.b_final;
        
        // Verificação de erro na predição
        if output.is_nan() || output.is_infinite() {
            Err(LSTMError::ForwardError("Model forward pass resulted in invalid value (NaN/Inf)".to_string()))
        } else {
            Ok(output)
        }
    }
}

// Função para normalizar os dados
fn normalize_data(data: &[f32], mean: f32, std: f32) -> Vec<f32> {
    // Adicionando uma pequena proteção contra divisão por zero
    let std_safe = std.max(1e-8);
    data.iter().map(|&x| (x - mean) / std_safe).collect()
}

// Função para criar sequências para predição (última sequência disponível)
fn create_prediction_sequence(
    data: &[StockData],
    seq_length: usize,
    closing_mean: f32,
    closing_std: f32,
    opening_mean: f32,
    opening_std: f32,
) -> Result<Vec<f32>, LSTMError> {
    if data.len() < seq_length {
        return Err(LSTMError::InsufficientData {
            asset: data.first().map(|d| d.date.clone()).unwrap_or("Unknown".to_string()),
            required: seq_length,
            actual: data.len(),
        });
    }

    let start_index = data.len() - seq_length;
    let closing_prices: Vec<f32> = data[start_index..].iter().map(|d| d.closing).collect();
    let opening_prices: Vec<f32> = data[start_index..].iter().map(|d| d.opening).collect();

    let norm_closing = normalize_data(&closing_prices, closing_mean, closing_std);
    let norm_opening = normalize_data(&opening_prices, opening_mean, opening_std);

    let mut sequence = Vec::new();
    for i in 0..seq_length {
        sequence.push(norm_closing[i]);
        sequence.push(norm_opening[i]);

        // Média móvel 5 períodos (simplificada para predição)
        if i >= 4 {
            let ma5 = closing_prices[(i-4)..=i].iter().sum::<f32>() / 5.0;
            sequence.push((ma5 - closing_mean) / closing_std.max(1e-8));
        } else {
            sequence.push(norm_closing[i]); // Preenche com o próprio fechamento se não houver dados suficientes
        }

        // Momento (diferença do fechamento do dia anterior)
        if i > 0 {
            sequence.push((closing_prices[i] - closing_prices[i-1]) / closing_std.max(1e-8));
        } else {
            sequence.push(0.0); // Zero para o primeiro elemento
        }
    }
    
    if data.len() >= 3 && seq_length >= 3 { // Log detalhado para as últimas entradas
        info!("  📈 Last 3 data points used for sequence creation:");
        for i in (seq_length - 3)..seq_length {
             info!("    Data[{}]: Date={}, Close={}, Open={}", start_index + i, data[start_index + i].date, data[start_index + i].closing, data[start_index + i].opening);
        }
        info!("  🧮 First 6 normalized features of the sequence: {:?}", &sequence[..6.min(sequence.len())]);
    }

    Ok(sequence)
}

// Função principal para gerar previsões para N dias
fn predict_prices(
    model_weights: ModelWeights, // Tomamos posse aqui
    data: Vec<StockData>,
    num_predictions: usize,
    verbose: bool,
) -> Result<Vec<(String, f32)>, LSTMError> {
    info!("  🧠 Starting prediction process...");
    let seq_length = model_weights.seq_length;
    let closing_mean = model_weights.closing_mean;
    let closing_std = model_weights.closing_std;
    let opening_mean = model_weights.opening_mean;
    let opening_std = model_weights.opening_std;

    info!("    Model Parameters - Seq Len: {}, Close Mean: {:.4}, Close Std: {:.4}", seq_length, closing_mean, closing_std);

    // Cria o modelo
    let model = MultiLayerLSTM::from_weights(&model_weights); // Passa a referência

    // Parse da última data
    let last_date_str = &data.last().ok_or_else(|| LSTMError::DataLoadingError("No data available".to_string()))?.date;
    info!("  📅 Last known data date: {}", last_date_str);
    // Tenta vários formatos de data com base nos logs do lstmfiletrain
    let last_date = NaiveDate::parse_from_str(last_date_str, "%d.%m.%Y")
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%Y-%m-%d"))
        .or_else(|_| NaiveDate::parse_from_str(last_date_str, "%m/%d/%Y"))
        .map_err(|e| LSTMError::DateParseError(format!("Failed to parse last date '{}': {}", last_date_str, e)))?;

    let mut predictions = Vec::new();
    let mut current_data = data; // Cópia para poder modificar durante as iterações

    for i in 0..num_predictions {
        info!("  🔮 Prediction Step {}/{}", i + 1, num_predictions);
        // Cria a sequência de entrada com base nos dados atuais
        let sequence = create_prediction_sequence(
            &current_data,
            seq_length,
            closing_mean,
            closing_std,
            opening_mean,
            opening_std,
        )?;

        // Faz a predição
        let pred_normalized = model.forward(&sequence)?;
        info!("    Normalized Prediction: {:.6}", pred_normalized);

        // Desnormaliza a predição
        let pred_price = (pred_normalized * closing_std) + closing_mean;
        info!("    Denormalized Prediction (R$): {:.2}", pred_price);

        // Calcula a data da previsão
        let pred_date = last_date + Duration::days((i + 1) as i64);
        let pred_date_str = pred_date.format("%d.%m.%Y").to_string();
        info!("    Predicted Date: {}", pred_date_str);

        predictions.push((pred_date_str.clone(), pred_price));

        if verbose {
            println!("  🔮 Prediction {}/{}: Date: {}, Price: R$ {:.2}", i + 1, num_predictions, pred_date_str, pred_price);
        }

        // --- Atualiza os dados para a próxima predição ---
        current_data.push(StockData {
            date: pred_date_str,
            closing: pred_price,
            opening: pred_price, // Simplificação
        });
        info!("  🔄 Updated data sequence with predicted point.");
    }

    Ok(predictions)
}

// Função para carregar um modelo específico com base no prefixo do nome do ativo
fn load_model_by_asset_prefix(client: &mut Client, asset_prefix: &str) -> Result<ModelWeights, LSTMError> {
    info!("  🔍 Searching for model with asset prefix: '{}'", asset_prefix);
    let query = "SELECT asset, weights_json FROM lstm_weights_v3 WHERE asset LIKE $1 || '%' AND source = $2 ORDER BY created_at DESC LIMIT 1";
    let search_pattern = format!("{}%", asset_prefix);

    let row_opt = client.query_opt(query, &[&search_pattern, &"investing"])?;

    if let Some(row) = row_opt {
        let full_asset_name: String = row.get("asset");
        info!("  ✅ Found model record for asset: {}", full_asset_name);
        // CORREÇÃO: Obter como String primeiro, depois deserializar
        let weights_json_str: String = row.get("weights_json");
        info!("  📦 Deserializing model weights from JSON...");
        let weights: ModelWeights = serde_json::from_str(&weights_json_str)
            .map_err(|e| LSTMError::DeserializationError(format!("Failed to deserialize model for {}: {}", full_asset_name, e)))?;
        info!("  ✅ Model successfully loaded and deserialized for asset: {}", full_asset_name);
        Ok(weights) // Retorna ModelWeights diretamente
    } else {
        Err(LSTMError::ModelNotFound(asset_prefix.to_string()))
    }
}

// Função para carregar todos os modelos
fn load_all_models(client: &mut Client) -> Result<Vec<(String, ModelWeights)>, LSTMError> {
    info!("  📡 Fetching all models from lstm_weights_v3 table...");
    let query = "SELECT asset, weights_json FROM lstm_weights_v3 WHERE source = $1 ORDER BY asset";
    let rows: Vec<Row> = client.query(query, &[&"investing"])?;

    info!("  📊 Found {} model records in the database", rows.len());
    let mut models = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let full_asset_name: String = row.get("asset");
        info!("  🔧 Processing model record {}/{}: {}", i+1, rows.len(), full_asset_name);
        // CORREÇÃO: Obter como String primeiro, depois deserializar
        let weights_json_str: String = row.get("weights_json");
        match serde_json::from_str::<ModelWeights>(&weights_json_str) {
            Ok(weights) => {
                models.push((full_asset_name.clone(), weights));
                info!("  ✅ Model {} successfully loaded and deserialized", full_asset_name);
            }
            Err(e) => {
                // Se um modelo não puder ser carregado, registra o erro mas continua com os outros
                error!("  ❌ Failed to deserialize model for asset {}: {}", full_asset_name, e);
            }
        }
    }

    if models.is_empty() && !rows.is_empty() {
        // Se havia linhas no DB mas nenhuma pôde ser desserializada
        warn!("⚠️ Models found in DB but none could be successfully loaded.");
    } else if models.is_empty() {
        warn!("⚠️ No models found in the database for source 'investing'.");
    } else {
        info!("  ✅ Successfully loaded {} models", models.len());
    }

    Ok(models)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(if cfg!(debug_assertions) { log::LevelFilter::Debug } else { log::LevelFilter::Info })
        .init();

    let args = Args::parse();
    println!("🚀 Starting LSTM Stock Price Prediction (Detailed Log)");
    println!("==================================================");

    match &args.asset {
        Some(asset_name) => {
            info!("🎯 Target Asset: {}", asset_name);
        },
        None => {
            info!("🎯 Target: ALL assets in the database");
        }
    }
    info!("🔢 Number of Predictions Requested: {}", args.num_predictions);
    info!("🔊 Verbose Mode: {}", args.verbose);
    info!("🔗 PostgreSQL Connection String: {}", mask_password(&args.pg_conn)); // Função auxiliar para mascarar senha

    info!("📡 Connecting to PostgreSQL database...");
    let mut pg_client = Client::connect(&args.pg_conn, NoTls)?;
    info!("✅ Successfully connected to PostgreSQL");

    let models_to_process: Vec<(String, ModelWeights)> = if let Some(asset_prefix) = &args.asset {
        // Carregar modelo para um ativo específico
        info!("📥 Loading model for specific asset '{}'...", asset_prefix);
        let model_weights = load_model_by_asset_prefix(&mut pg_client, asset_prefix)?;
        vec![(model_weights.asset.clone(), model_weights)] // Armazena com o nome completo
    } else {
        // Carregar modelos para todos os ativos
        info!("📥 Loading models for ALL assets...");
        load_all_models(&mut pg_client)?
    };

    if models_to_process.is_empty() {
        if args.asset.is_some() {
             error!("❌ No model found for the specified asset prefix '{}'.", args.asset.as_ref().unwrap());
             println!("\n❌ Prediction process failed: Model not found for '{}'", args.asset.as_ref().unwrap());
        } else {
             warn!("⚠️ No models found in the database. Nothing to predict.");
             println!("\n⚠️ Prediction process completed: No models found in the database.");
        }
        return Ok(());
    }

    info!("📊 Starting prediction loop for {} asset(s)...", models_to_process.len());
    println!("\n🔮 Starting Predictions");
    println!("======================");

    let mut successful_predictions = 0;
    let mut failed_predictions = 0;

    // --- Loop de Predição ---
    for (full_asset_name, model_weights) in models_to_process {
        println!("\n--- 📊 Processing Asset: {} ---", full_asset_name);
        info!("📈 Beginning prediction process for asset: {}", full_asset_name);
       
        // 1. Carregar dados recentes para o ativo
        info!("📂 Loading recent historical data for {} from PostgreSQL...", full_asset_name);
        // CORRIGIDA: Passando args.verbose como parâmetro
        let data_result = load_data_from_postgres(&mut pg_client, &full_asset_name, args.verbose);
        let data = match data_result {
            Ok(d) => d,
            Err(e) => {
                error!("❌ Failed to load data for {}: {}", full_asset_name, e);
                failed_predictions += 1;
                continue;
            }
        };

        if data.is_empty() {
            warn!("⚠️ No data loaded for {}. Skipping prediction.", full_asset_name);
            failed_predictions += 1;
            continue;
        }

        // 2. Gerar previsões
        info!("🔮 Generating {} predictions for {}...", args.num_predictions, full_asset_name);
        match predict_prices(model_weights, data, args.num_predictions, args.verbose) {
            Ok(predictions) => {
                info!("✅ Predictions successfully generated for {}", full_asset_name);
                successful_predictions += 1;

                // 3. Mostrar resultados finais
                println!("\n--- 🎯 Final Predictions for {} ---", full_asset_name);
                for (date, price) in &predictions {
                    println!("📅 {}: R$ {:.2}", date, price);
                }
                println!("--- End of predictions for {} ---", full_asset_name);
            }
            Err(e) => {
                error!("❌ Failed to generate predictions for {}: {}", full_asset_name, e);
                failed_predictions += 1;
                // Continua com o próximo ativo em caso de erro
            }
        }
    }

    println!("\n🏁 Prediction Process Summary");
    println!("==============================");
    println!("✅ Successful Assets: {}", successful_predictions);
    println!("❌ Failed Assets: {}", failed_predictions);
    println!("📊 Total Assets Processed: {}", successful_predictions + failed_predictions);
    println!("\n🎉 Prediction process completed!");
    Ok(())
}

// Função auxiliar simples para mascarar a senha na string de conexão (para log)
fn mask_password(conn_str: &str) -> String {
    // Esta é uma implementação muito básica e pode não funcionar para todos os casos.
    // Para produção, considere usar um crate dedicado como `url`.
    if let Some(at_pos) = conn_str.find('@') {
        if let Some(colon_pos) = conn_str[..at_pos].find(':') {
            let start = conn_str[..colon_pos].to_string();
            let end = conn_str[at_pos..].to_string();
            format!("{}:***{}", start, end)
        } else {
            conn_str.to_string()
        }
    } else {
        conn_str.to_string()
    }
}



// Para um ativo específico (com correção) : 
// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose
// cargo run --release -- --asset SLCE3 --num-predictions 3 # Modo resumido

// Para todos os ativos salvos : 
// cargo run --release -- --num-predictions 5 --verbose
// cargo run --release -- --num-predictions 3 # Modo resumido



// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose


// cargo run --release -- --asset SLCE3 --num-predictions 5 --verbose
 
// cargo run --release -- --seq-length 40 --hidden-size 64 --num-layers 2 --epochs 50 --batch-size 16 --dropout-rate 0.3 --learning-rate 0.0005
 

// cargo run --release -- --asset "SLCE3 Dados Históricos" --num-predictions 5 --verbose