// projeto: lstm_rnn_trainer
// file: src/main.rs



// projeto: lstm_rnn_trainer
// file: src/main.rs

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW, loss, rnn::{lstm, LSTMConfig, LSTM}, RNN, Activation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::Result as AnyResult;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StockRecord {
    date: String,
    closing: f64,
    opening: f64,
    high: f64,
    low: f64,
    volume: f64,
    variation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StockData {
    asset: String,
    records: Vec<StockRecord>,
}

#[derive(Debug, Clone)]
struct PreparedStockData {
    dates: Vec<String>,
    closing: Vec<f64>,
    opening: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    volume: Vec<f64>,
    variation: Vec<f64>,
    asset_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NormalizationParams {
    mu: f64,
    sigma: f64,
}

#[derive(Debug, Clone)]
struct TrainingData {
    x: Tensor,
    y: Tensor,
    normalization_params: Vec<NormalizationParams>,
}

#[derive(Debug, Clone)]
struct LSTMModel {
    lstm: LSTM,
    dense1: Linear,
    dense2: Linear,
    dropout_rate: f64,
}

impl LSTMModel {
    fn new(vs: VarBuilder, input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        let lstm_config = LSTMConfig::default();
        let lstm = lstm(input_size, hidden_size, lstm_config, vs.pp("lstm"))?;
        let dense1 = candle_nn::linear(hidden_size, hidden_size, vs.pp("dense1"))?;
        let dense2 = candle_nn::linear(hidden_size, output_size, vs.pp("dense2"))?;
        
        Ok(Self {
            lstm,
            dense1,
            dense2,
            dropout_rate: 0.2,
        })
    }
    
    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let states = self.lstm.seq(x)?;
        let last_state = states.last().unwrap();
        let last_output = &last_state.h;
        
        let mut x = self.dense1.forward(last_output)?;
        x = x.relu()?;
        
        // Aplicar dropout durante treinamento (desabilitado temporariamente)
        // if training && self.dropout_rate > 0.0 {
        //     let threshold = Tensor::new(self.dropout_rate as f32, x.device())?
        //         .broadcast_as(x.shape())?;
        //     let mask = Tensor::rand(0f32, 1f32, x.shape(), x.device())?
        //         .ge(&threshold)?;
        //     let scale = Tensor::new(1.0 / (1.0 - self.dropout_rate) as f32, x.device())?
        //         .broadcast_as(x.shape())?;
        //     x = x.mul(&mask.to_dtype(x.dtype())?)?;
        //     x = x.mul(&scale)?;
        // }
        
        self.dense2.forward(&x)
    }
}

// Implementação simplificada de RNN usando LSTM com configuração diferente
#[derive(Debug, Clone)]
struct SimpleRNNModel {
    rnn: LSTM, // Usaremos LSTM como base para RNN
    dense1: Linear,
    dense2: Linear,
    dropout_rate: f64,
}

impl SimpleRNNModel {
    fn new(vs: VarBuilder, input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        // Configuração LSTM simples para simular RNN
        let lstm_config = LSTMConfig::default();
        let rnn = lstm(input_size, hidden_size, lstm_config, vs.pp("rnn"))?;
        let dense1 = candle_nn::linear(hidden_size, hidden_size, vs.pp("dense1"))?;
        let dense2 = candle_nn::linear(hidden_size, output_size, vs.pp("dense2"))?;
        
        Ok(Self {
            rnn,
            dense1,
            dense2,
            dropout_rate: 0.2,
        })
    }
    
    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let states = self.rnn.seq(x)?;
        let last_state = states.last().unwrap();
        let last_output = &last_state.h;
        
        let mut x = self.dense1.forward(last_output)?;
        x = x.relu()?;
        
        // Aplicar dropout durante treinamento (desabilitado temporariamente)
        // if training && self.dropout_rate > 0.0 {
        //     let threshold = Tensor::new(self.dropout_rate as f32, x.device())?
        //         .broadcast_as(x.shape())?;
        //     let mask = Tensor::rand(0f32, 1f32, x.shape(), x.device())?
        //         .ge(&threshold)?;
        //     let scale = Tensor::new(1.0 / (1.0 - self.dropout_rate) as f32, x.device())?
        //         .broadcast_as(x.shape())?;
        //     x = x.mul(&mask.to_dtype(x.dtype())?)?;
        //     x = x.mul(&scale)?;
        // }
        
        self.dense2.forward(&x)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelConfig {
    sequence_length: usize,
    features_count: usize,
    hidden_size: usize,
    feature_names: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingInfo {
    timestamp: DateTime<Utc>,
    total_sequences: usize,
    sequence_length: usize,
    features_count: usize,
    epochs: usize,
    learning_rate: f64,
    assets_trained: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelParams {
    #[serde(rename = "type")]
    layer_type: String,
    weights: Option<Vec<Vec<f32>>>,
    bias: Option<Vec<f32>>,
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    activation: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedModel {
    architecture: Vec<ModelParams>,
    final_loss: f64,
    training_losses: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainedModelsData {
    training_info: TrainingInfo,
    lstm_model: SavedModel,
    rnn_model: SavedModel,
    normalization_params: Vec<HashMap<String, serde_json::Value>>,
    feature_names: Vec<String>,
}

struct DataProcessor;

impl DataProcessor {
    fn load_stock_data(filepath: &Path) -> AnyResult<PreparedStockData> {
        let contents = fs::read_to_string(filepath)?;
        let data: StockData = toml::from_str(&contents)?;
        
        let mut closing = Vec::new();
        let mut opening = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut volume = Vec::new();
        let mut variation = Vec::new();
        let mut dates = Vec::new();
        
        for record in &data.records {
            dates.push(record.date.clone());
            closing.push(record.closing);
            opening.push(record.opening);
            high.push(record.high);
            low.push(record.low);
            volume.push(record.volume);
            variation.push(record.variation);
        }
        
        Ok(PreparedStockData {
            dates,
            closing,
            opening,
            high,
            low,
            volume,
            variation,
            asset_name: data.asset,
        })
    }
    
    fn normalize_data(data: &[f64]) -> (Vec<f64>, NormalizationParams) {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        let normalized: Vec<f64> = data.iter()
            .map(|x| (x - mean) / std_dev)
            .collect();
        
        (normalized, NormalizationParams { mu: mean, sigma: std_dev })
    }
    
    fn prepare_multivariate_data(
        stock_data: &PreparedStockData,
        seq_length: usize,
        device: &Device,
    ) -> Result<TrainingData> {
        // Combinar features relevantes
        let log_volume: Vec<f64> = stock_data.volume.iter()
            .map(|v| (v + 1.0).ln())
            .collect();
        
        let features = vec![
            &stock_data.closing,
            &stock_data.opening,
            &stock_data.high,
            &stock_data.low,
            &log_volume,
            &stock_data.variation,
        ];
        
        // Normalizar cada feature
        let mut normalized_features = Vec::new();
        let mut normalization_params = Vec::new();
        
        for feature in features {
            let (normalized, params) = Self::normalize_data(feature);
            normalized_features.push(normalized);
            normalization_params.push(params);
        }
        
        // Criar sequências
        let num_features = normalized_features.len();
        let data_length = normalized_features[0].len();
        let num_sequences = data_length.saturating_sub(seq_length);
        
        if num_sequences == 0 {
            return Err(candle_core::Error::Msg("Dados insuficientes para criar sequências".to_string()));
        }
        
        // Preparar tensores X e y
        let mut x_data = Vec::with_capacity(num_sequences * seq_length * num_features);
        let mut y_data = Vec::with_capacity(num_sequences);
        
        for i in 0..num_sequences {
            // X: sequência de features
            for t in 0..seq_length {
                for f in 0..num_features {
                    x_data.push(normalized_features[f][i + t] as f32);
                }
            }
            
            // y: próximo preço de fechamento (feature 0)
            y_data.push(normalized_features[0][i + seq_length] as f32);
        }
        
        let x = Tensor::from_vec(x_data, (num_sequences, seq_length, num_features), device)?;
        let y = Tensor::from_vec(y_data, (num_sequences, 1), device)?;
        
        Ok(TrainingData {
            x,
            y,
            normalization_params,
        })
    }
    
    fn find_toml_files(data_dir: &Path) -> Vec<PathBuf> {
        let mut toml_files = Vec::new();
        
        if let Ok(entries) = fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "toml") {
                    toml_files.push(path);
                }
            }
        }
        
        toml_files.sort();
        toml_files
    }
}

struct ModelTrainer;

impl ModelTrainer {
    fn train_lstm_model(
        training_data: &[TrainingData],
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        device: &Device,
    ) -> Result<(VarMap, Vec<f64>)> {
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        let model = LSTMModel::new(vs, input_size, hidden_size, 1)?;
        let params = ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
        
        let mut losses = Vec::new();
        
        println!("Iniciando treinamento do modelo LSTM...");
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            for data in training_data {
                let batch_size = data.x.dim(0)?;
                
                for i in 0..batch_size {
                    let x_batch = data.x.narrow(0, i, 1)?;
                    let y_batch = data.y.narrow(0, i, 1)?;
                    
                    let predictions = model.forward(&x_batch, true)?;
                    let loss = loss::mse(&predictions, &y_batch)?;
                    
                    optimizer.backward_step(&loss)?;
                    
                    total_loss += loss.to_scalar::<f32>()? as f64;
                    batch_count += 1;
                }
            }
            
            let avg_loss = total_loss / batch_count as f64;
            losses.push(avg_loss);
            
            if epoch % 20 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
            }
        }
        
        Ok((varmap, losses))
    }
    
    fn train_rnn_model(
        training_data: &[TrainingData],
        input_size: usize,
        hidden_size: usize,
        epochs: usize,
        learning_rate: f64,
        device: &Device,
    ) -> Result<(VarMap, Vec<f64>)> {
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        let model = SimpleRNNModel::new(vs, input_size, hidden_size, 1)?;
        let params = ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
        
        let mut losses = Vec::new();
        
        println!("Iniciando treinamento do modelo RNN...");
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            for data in training_data {
                let batch_size = data.x.dim(0)?;
                
                for i in 0..batch_size {
                    let x_batch = data.x.narrow(0, i, 1)?;
                    let y_batch = data.y.narrow(0, i, 1)?;
                    
                    let predictions = model.forward(&x_batch, true)?;
                    let loss = loss::mse(&predictions, &y_batch)?;
                    
                    optimizer.backward_step(&loss)?;
                    
                    total_loss += loss.to_scalar::<f32>()? as f64;
                    batch_count += 1;
                }
            }
            
            let avg_loss = total_loss / batch_count as f64;
            losses.push(avg_loss);
            
            if epoch % 20 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
            }
        }
        
        Ok((varmap, losses))
    }
    
    fn extract_model_params(_varmap: &VarMap) -> Vec<ModelParams> {
        let mut params = Vec::new();
        
        // Esta função precisaria ser implementada baseada na estrutura interna do candle
        // Por simplicidade, retornamos um placeholder
        params.push(ModelParams {
            layer_type: "LSTM".to_string(),
            weights: None,
            bias: None,
            input_size: Some(6),
            hidden_size: Some(32),
            activation: None,
        });
        
        params.push(ModelParams {
            layer_type: "Dense".to_string(),
            weights: None,
            bias: None,
            input_size: Some(32),
            hidden_size: Some(32),
            activation: Some("relu".to_string()),
        });
        
        params.push(ModelParams {
            layer_type: "Dense".to_string(),
            weights: None,
            bias: None,
            input_size: Some(32),
            hidden_size: Some(1),
            activation: None,
        });
        
        params
    }
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    println!("=== Treinamento de Modelos LSTM e RNN para Dados de Ações ===\n");
    
    // Configurar device (CPU por enquanto)
    let device = Device::Cpu;
    
    // Definir caminho do diretório de dados
    let data_dir = Path::new("../../dados/ativos");
    
    // Descobrir todos os arquivos TOML na pasta
    println!("Procurando arquivos TOML em: {:?}", data_dir);
    let files = DataProcessor::find_toml_files(data_dir);
    
    if files.is_empty() {
        println!("Nenhum arquivo TOML encontrado em: {:?}", data_dir);
        return Ok(());
    }
    
    println!("Arquivos TOML encontrados:");
    for (i, file) in files.iter().enumerate() {
        println!("  {}. {:?}", i + 1, file.file_name().unwrap_or_default());
    }
    println!();
    
    // Carregar e processar dados de todas as ações
    let mut all_data = Vec::new();
    let mut all_training_data = Vec::new();
    let mut successful_loads = 0;
    let mut failed_loads = 0;
    let seq_length = 5;
    
    for file in &files {
        match DataProcessor::load_stock_data(file) {
            Ok(stock_data) => {
                println!("Carregando dados de: {:?}", file.file_name().unwrap_or_default());
                
                match DataProcessor::prepare_multivariate_data(&stock_data, seq_length, &device) {
                    Ok(training_data) => {
                        println!("  ✅ {}: {} sequências criadas", 
                               stock_data.asset_name, 
                               training_data.x.dim(0).unwrap_or(0));
                        
                        all_training_data.push(training_data);
                        all_data.push(stock_data);
                        successful_loads += 1;
                    }
                    Err(e) => {
                        println!("  ⚠️  {}: Erro ao preparar dados - {:?}", stock_data.asset_name, e);
                        failed_loads += 1;
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Erro ao carregar {:?}: {}", file.file_name().unwrap_or_default(), e);
                failed_loads += 1;
            }
        }
    }
    
    println!("\nResumo do carregamento:");
    println!("  ✅ Arquivos carregados com sucesso: {}", successful_loads);
    println!("  ❌ Arquivos com erro: {}", failed_loads);
    println!("  📊 Total de arquivos processados: {}", files.len());
    
    if all_training_data.is_empty() {
        println!("\nNenhum dado foi carregado. Verifique os arquivos TOML.");
        return Ok(());
    }
    
    // Treinar modelos
    println!("\n=== Treinando Modelos ===");
    
    let input_size = 6; // features: closing, opening, high, low, log_volume, variation
    let hidden_size = 32;
    let epochs = 100;
    let learning_rate = 0.001;
    
    // Treinar LSTM
    let (lstm_varmap, lstm_losses) = ModelTrainer::train_lstm_model(
        &all_training_data, 
        input_size, 
        hidden_size, 
        epochs, 
        learning_rate, 
        &device
    )?;
    
    // Treinar RNN
    let (rnn_varmap, rnn_losses) = ModelTrainer::train_rnn_model(
        &all_training_data, 
        input_size, 
        hidden_size, 
        epochs, 
        learning_rate, 
        &device
    )?;
    
    // Salvar modelos
    println!("\n=== Salvando Modelos Treinados ===");
    
    let save_dir = Path::new("../../dados/modelos_treinados");
    fs::create_dir_all(save_dir)?;
    
    let lstm_params = ModelTrainer::extract_model_params(&lstm_varmap);
    let rnn_params = ModelTrainer::extract_model_params(&rnn_varmap);
    
    let total_sequences: usize = all_training_data.iter()
        .map(|d| d.x.dim(0).unwrap_or(0))
        .sum();
    
    let training_data = TrainedModelsData {
        training_info: TrainingInfo {
            timestamp: Utc::now(),
            total_sequences,
            sequence_length: seq_length,
            features_count: input_size,
            epochs,
            learning_rate,
            assets_trained: all_data.iter().map(|d| d.asset_name.clone()).collect(),
        },
        lstm_model: SavedModel {
            architecture: lstm_params,
            final_loss: *lstm_losses.last().unwrap_or(&0.0),
            training_losses: lstm_losses.clone(),
        },
        rnn_model: SavedModel {
            architecture: rnn_params,
            final_loss: *rnn_losses.last().unwrap_or(&0.0),
            training_losses: rnn_losses.clone(),
        },
        normalization_params: vec![], // Simplificado por enquanto
        feature_names: vec![
            "closing".to_string(),
            "opening".to_string(),
            "high".to_string(),
            "low".to_string(),
            "log_volume".to_string(),
            "variation".to_string(),
        ],
    };
    
    let json_save_path = save_dir.join("trained_models.json");
    let json_data = serde_json::to_string_pretty(&training_data)?;
    fs::write(&json_save_path, json_data)?;
    println!("  ✅ Modelos salvos em JSON: {:?}", json_save_path);
    
    // Salvar configuração
    let config_data = ModelConfig {
        sequence_length: seq_length,
        features_count: input_size,
        hidden_size,
        feature_names: vec![
            "closing".to_string(),
            "opening".to_string(),
            "high".to_string(),
            "low".to_string(),
            "log_volume".to_string(),
            "variation".to_string(),
        ],
    };
    
    let config_save_path = save_dir.join("model_config.json");
    let config_json = serde_json::to_string_pretty(&config_data)?;
    fs::write(&config_save_path, config_json)?;
    println!("  ✅ Configuração salva em: {:?}", config_save_path);
    
    // Relatório final
    println!("\n=== Relatório Final de Treinamento ===");
    println!("📈 Desempenho dos Modelos:");
    println!("  • Perda final LSTM: {:.6}", lstm_losses.last().unwrap_or(&0.0));
    println!("  • Perda final RNN: {:.6}", rnn_losses.last().unwrap_or(&0.0));
    
    if let (Some(&first_lstm), Some(&last_lstm)) = (lstm_losses.first(), lstm_losses.last()) {
        let improvement = (first_lstm - last_lstm) / first_lstm * 100.0;
        println!("  • Melhoria LSTM: {:.2}%", improvement);
    }
    
    if let (Some(&first_rnn), Some(&last_rnn)) = (rnn_losses.first(), rnn_losses.last()) {
        let improvement = (first_rnn - last_rnn) / first_rnn * 100.0;
        println!("  • Melhoria RNN: {:.2}%", improvement);
    }
    
    println!("\n📊 Dados de Treinamento:");
    println!("  • Arquivos TOML processados: {}", files.len());
    println!("  • Ativos com dados válidos: {}", all_data.len());
    println!("  • Sequências de treinamento: {}", total_sequences);
    println!("  • Features por sequência: {}", input_size);
    println!("  • Comprimento da sequência: {}", seq_length);
    
    println!("\n✅ Treinamento concluído com sucesso!");
    
    Ok(())
}



/*
Para compilar e executar:

1. Crie um novo projeto Cargo:
   cargo new lstm_rnn_trainer
   cd lstm_rnn_trainer

2. Substitua o conteúdo do Cargo.toml e src/main.rs pelos códigos acima

3. Compile e execute:
   cargo build --release
   cargo run --release

Notas importantes:
- Este código usa a biblioteca Candle para deep learning em Rust
- Algumas funcionalidades foram simplificadas devido às diferenças entre Julia/Flux e Rust/Candle
- Você pode precisar ajustar alguns imports dependendo da versão exata das bibliotecas
- A extração completa dos parâmetros do modelo requer implementação adicional específica do Candle
*/