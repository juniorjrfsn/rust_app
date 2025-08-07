// projeto: lstmrnntrain
// file: src/main.rs
// Main entry point for multi-model deep learning training system



// projeto: lstmrnntrain
// file: src/main.rs
// Sistema principal de treinamento de redes neurais profundas para previsão financeira

mod neural;

use clap::Parser;
use chrono::Utc;
use log::{info, warn, error, debug};
use ndarray::Array2;
use std::time::Instant;

use crate::neural::data::{connect_db, ensure_tables_exist, DataLoader};
use crate::neural::metrics::{TrainingMetrics, MetricsTracker};
use crate::neural::model::{ModelType, NeuralNetwork};
use crate::neural::storage::save_model_to_postgres;
use crate::neural::utils::{AdamOptimizer, TrainingError, LearningRateScheduler};

/// Limpa e normaliza nomes de ativos removendo sufixos comuns
fn clean_asset_name(asset: &str) -> String {
    let cleaned = asset
        .replace(" Dados Históricos", "")
        .replace(" Historical Data", "")
        .replace(" Preços Históricos", "")
        .replace(" - Investing.com", "")
        .replace("(", "")
        .replace(")", "")
        .trim()
        .to_string();
    
    // Limitar a 80 caracteres para compatibilidade com banco
    if cleaned.len() > 80 {
        cleaned[..80].to_string()
    } else {
        cleaned
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "lstmrnntrain",
    author = "AI Trading Systems",
    version = "2.0.0",
    about = "Sistema de treinamento de redes neurais para previsão de preços de ativos financeiros",
    long_about = "Sistema avançado que suporta múltiplos modelos (LSTM, RNN, MLP, CNN) para análise e previsão de séries temporais financeiras com otimizações de performance e regularização."
)]
struct Cli {
    /// Ativo específico para treinar (ex: PETR4, VALE3)
    #[arg(long, default_value_t = String::from("PETR4"))]
    asset: String,

    /// Treinar modelos para todos os ativos disponíveis
    #[arg(long)]
    all_assets: bool,

    /// Tipo de modelo de rede neural
    #[arg(long, value_enum, default_value = "lstm")]
    model_type: ModelTypeArg,

    /// Comprimento da sequência de entrada
    #[arg(long, default_value_t = 40, help = "Número de dias históricos para predição")]
    seq_length: usize,

    /// Tamanho da camada oculta
    #[arg(long, default_value_t = 128, help = "Número de neurônios nas camadas ocultas")]
    hidden_size: usize,

    /// Número de camadas do modelo
    #[arg(long, default_value_t = 3, help = "Número de camadas profundas")]
    num_layers: usize,

    /// Número de épocas de treinamento
    #[arg(long, default_value_t = 100, help = "Iterações completas sobre o dataset")]
    epochs: usize,

    /// Tamanho do lote para treinamento
    #[arg(long, default_value_t = 64, help = "Número de amostras por lote")]
    batch_size: usize,

    /// Taxa de dropout para regularização
    #[arg(long, default_value_t = 0.3, help = "Taxa de dropout (0.0-1.0)")]
    dropout_rate: f64,

    /// Taxa de aprendizado inicial
    #[arg(long, default_value_t = 0.001, help = "Learning rate para Adam optimizer")]
    learning_rate: f64,

    /// Peso da regularização L2
    #[arg(long, default_value_t = 0.01, help = "Peso da regularização L2")]
    l2_weight: f64,

    /// Norma máxima para gradient clipping
    #[arg(long, default_value_t = 1.0, help = "Máximo para gradient clipping")]
    clip_norm: f64,

    /// Early stopping patience
    #[arg(long, default_value_t = 15, help = "Épocas sem melhoria para parar")]
    patience: usize,

    /// Razão de divisão treino/validação
    #[arg(long, default_value_t = 0.8, help = "Proporção dos dados para treinamento")]
    train_split: f64,

    /// URL de conexão com PostgreSQL
    #[arg(long, default_value_t = String::from("postgresql://postgres:postgres@localhost:5432/rnn_db"))]
    db_url: String,

    /// Modo verboso de logging
    #[arg(long)]
    verbose: bool,

    /// Salvar checkpoints periodicamente
    #[arg(long, default_value_t = 10, help = "Frequência de salvamento (épocas)")]
    save_freq: usize,

    /// Usar scheduler de learning rate
    #[arg(long)]
    use_scheduler: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum ModelTypeArg {
    Lstm,
    Rnn,
    Mlp,
    Cnn,
}

impl From<ModelTypeArg> for ModelType {
    fn from(arg: ModelTypeArg) -> Self {
        match arg {
            ModelTypeArg::Lstm => ModelType::LSTM,
            ModelTypeArg::Rnn => ModelType::RNN,
            ModelTypeArg::Mlp => ModelType::MLP,
            ModelTypeArg::Cnn => ModelType::CNN,
        }
    }
}

fn main() -> Result<(), TrainingError> {
    let cli = Cli::parse();
    
    // Configurar logging
    setup_logging(cli.verbose);
    
    let start_time = Instant::now();
    let model_type = ModelType::from(cli.model_type.clone());
    
    info!("🚀 Sistema de Deep Learning iniciado");
    info!("📊 Modelo: {:?} | Sequência: {} | Hidden: {} | Camadas: {}", 
          model_type, cli.seq_length, cli.hidden_size, cli.num_layers);
    info!("🕐 Iniciado em: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"));

    // Verificar e configurar banco de dados
    setup_database(&cli.db_url)?;

    let result = if cli.all_assets {
        train_all_assets(&cli, model_type)
    } else {
        let clean_asset = clean_asset_name(&cli.asset);
        info!("🎯 Treinando ativo único: {} (limpo: {})", cli.asset, clean_asset);
        train_single_asset(&cli, &clean_asset, model_type)
    };

    let elapsed = start_time.elapsed();
    match result {
        Ok(_) => {
            info!("✅ Treinamento concluído com sucesso em {:.2}s", elapsed.as_secs_f64());
            info!("🏁 Finalizado em: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"));
        }
        Err(e) => {
            error!("❌ Erro durante treinamento: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn setup_logging(verbose: bool) {
    let level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::from_default_env()
        .filter_level(level)
        .format_timestamp_secs()
        .init();
}

fn setup_database(db_url: &str) -> Result<(), TrainingError> {
    info!("🔧 Configurando banco de dados");
    let mut client = connect_db(db_url)?;
    ensure_tables_exist(&mut client)?;
    info!("✅ Banco de dados configurado");
    Ok(())
}

fn train_all_assets(cli: &Cli, model_type: ModelType) -> Result<(), TrainingError> {
    info!("🔄 Iniciando treinamento para todos os ativos");
    
    // Carregar lista de ativos
    let assets = {
        let mut client = connect_db(&cli.db_url)?;
        let mut loader = DataLoader::new(&mut client)?;
        loader.load_all_assets()?
    };
    
    info!("📋 Encontrados {} ativos únicos", assets.len());
    
    let mut successful_trains = 0;
    let mut failed_trains = 0;
    
    for (idx, asset) in assets.iter().enumerate() {
        let clean_asset = clean_asset_name(asset);
        info!("📌 Processando [{}/{}]: {} → {}", 
              idx + 1, assets.len(), asset, clean_asset);
        
        match train_single_asset(cli, &clean_asset, model_type) {
            Ok(_) => {
                successful_trains += 1;
                info!("✅ Sucesso para {}", clean_asset);
            }
            Err(e) => {
                failed_trains += 1;
                warn!("❌ Falha para {}: {}", clean_asset, e);
                continue;
            }
        }
    }
    
    info!("📊 Resumo do treinamento:");
    info!("   ├── Sucessos: {}", successful_trains);
    info!("   ├── Falhas: {}", failed_trains);
    info!("   └── Total: {}", assets.len());
    
    Ok(())
}

fn train_single_asset(
    cli: &Cli,
    asset: &str,
    model_type: ModelType,
) -> Result<(), TrainingError> {
    info!("🔧 Inicializando treinamento para: {}", asset);
    let training_start = Instant::now();
    
    // Conectar ao banco e carregar dados
    let mut client = connect_db(&cli.db_url)?;
    let mut loader = DataLoader::new(&mut client)?;
    
    debug!("📥 Carregando dados históricos");
    let records = loader.load_asset_data(asset)?;
    info!("✅ Carregados {} registros históricos", records.len());

    // Validar quantidade mínima de dados
    let min_required = cli.seq_length + 50; // Margem de segurança
    if records.len() < min_required {
        return Err(TrainingError::DataProcessing(
            format!("Dados insuficientes: {} registros, necessário pelo menos {}", 
                    records.len(), min_required)
        ));
    }

    // Criar sequências temporais
    debug!("🔧 Criando sequências de tamanho {}", cli.seq_length);
    let (train_seqs, train_targets, feature_stats) = 
        loader.create_sequences(&records, cli.seq_length)?;
    info!("✅ Criadas {} sequências temporais", train_seqs.len());

    // Pré-processamento e divisão dos dados
    debug!("🔄 Pré-processando dados");
    let (train_data, val_data) = preprocess_data(
        train_seqs, train_targets, &feature_stats, cli.train_split
    )?;
    
    info!("✅ Dados divididos - Treino: {} | Validação: {}", 
          train_data.0.len(), val_data.0.len());

    // Inicializar modelo
    info!("🛠️ Inicializando modelo {:?}", model_type);
    let mut model = NeuralNetwork::new(
        model_type,
        feature_stats.feature_names.len(),
        cli.hidden_size,
        cli.num_layers,
        cli.dropout_rate,
    )?;
    info!("✅ Modelo criado com {} parâmetros", model.num_parameters());

    // Configurar otimizador e scheduler
    let mut optimizer = AdamOptimizer::new(cli.learning_rate, 0.9, 0.999, 1e-8);
    
    let scheduler = if cli.use_scheduler {
        Some(LearningRateScheduler::StepDecay {
            initial_rate: cli.learning_rate,
            decay_rate: 0.5,
            step_size: cli.epochs / 4,
        })
    } else {
        None
    };

    // Inicializar tracker de métricas
    let mut metrics_tracker = MetricsTracker::new();
    
    // Loop de treinamento
    info!("🎓 Iniciando treinamento por {} épocas", cli.epochs);
    let mut best_val_loss = f64::INFINITY;
    
    for epoch in 1..=cli.epochs {
        let epoch_start = Instant::now();
        
        // Atualizar learning rate se necessário
        if let Some(ref sched) = scheduler {
            let new_lr = sched.get_rate(epoch - 1);
            optimizer.set_learning_rate(new_lr);
            debug!("📈 Learning rate atualizado para: {:.6}", new_lr);
        }

        // Passo de treinamento
        let train_loss = model.train_step(
            &train_data.0,
            &train_data.1,
            &mut optimizer,
            cli.batch_size,
            cli.l2_weight,
            cli.clip_norm,
        )?;

        // Validação
        let (val_loss, val_metrics) = model.validate(&val_data.0, &val_data.1)?;
        
        let epoch_time = epoch_start.elapsed().as_secs_f64();
        
        // Logging de progresso
        if epoch % 5 == 0 || epoch <= 10 {
            info!("📈 Época {}/{}: Train={:.6} | Val={:.6} | RMSE={:.6} | R²={:.4} | {:.1}s", 
                  epoch, cli.epochs, train_loss, val_loss, val_metrics.rmse, 
                  val_metrics.r_squared, epoch_time);
        }

        // Criar métricas para tracker
        let metrics = TrainingMetrics {
            asset: asset.to_string(),
            model_type: format!("{:?}", model_type),
            source: "database".to_string(),
            epoch,
            train_loss,
            val_loss,
            rmse: val_metrics.rmse,
            mae: val_metrics.mae,
            mape: val_metrics.mape,
            directional_accuracy: val_metrics.directional_accuracy,
            r_squared: val_metrics.r_squared,
            timestamp: Utc::now().to_rfc3339(),
        };

        // Early stopping check
        let should_stop = metrics_tracker.add_metrics(metrics.clone(), cli.patience);
        
        // Salvar modelo se melhorou ou na frequência especificada
        if val_loss < best_val_loss || epoch % cli.save_freq == 0 {
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                info!("🎯 Nova melhor perda de validação: {:.6}", best_val_loss);
            }
            
            debug!("💾 Salvando modelo na época {}", epoch);
            save_model_and_metrics(
                &cli.db_url, 
                &model, 
                asset, 
                model_type,
                &feature_stats,
                cli.seq_length,
                train_loss,
                val_loss,
                &val_metrics,
                epoch
            )?;
        }

        // Early stopping
        if should_stop {
            info!("⏹️ Early stopping acionado na época {}", epoch);
            break;
        }
    }

    let training_time = training_start.elapsed().as_secs_f64();
    
    // Resumo final
    metrics_tracker.print_summary();
    info!("⏱️ Treinamento de '{}' concluído em {:.2}s", asset, training_time);
    
    Ok(())
}

fn preprocess_data(
    train_seqs: Vec<Array2<f32>>,
    train_targets: Vec<f32>,
    feature_stats: &crate::neural::data::FeatureStats,
    split_ratio: f64,
) -> Result<((Vec<Array2<f64>>, Vec<f64>), (Vec<Array2<f64>>, Vec<f64>)), TrainingError> {
    
    debug!("🔄 Convertendo dados para f64 e normalizando");
    
    // Converter para f64 e normalizar
    let mut train_seqs: Vec<Array2<f64>> = train_seqs
        .iter()
        .map(|seq| {
            let mut seq_f64 = seq.mapv(|x| x as f64);
            normalize_sequence(&mut seq_f64, feature_stats);
            seq_f64
        })
        .collect();
        
    let mut train_targets: Vec<f64> = train_targets.iter().map(|&x| x as f64).collect();

    // Normalizar targets
    let closing_mean = feature_stats.closing_mean as f64;
    let closing_std = feature_stats.closing_std as f64;
    
    for target in &mut train_targets {
        *target = (*target - closing_mean) / closing_std;
    }

    // Dividir dados
    let split_index = (train_seqs.len() as f64 * split_ratio) as usize;
    let val_seqs: Vec<Array2<f64>> = train_seqs.split_off(split_index);
    let val_targets: Vec<f64> = train_targets.split_off(split_index);

    debug!("✅ Dados normalizados e divididos");
    Ok(((train_seqs, train_targets), (val_seqs, val_targets)))
}

fn normalize_sequence(seq: &mut Array2<f64>, feature_stats: &crate::neural::data::FeatureStats) {
    for i in 0..seq.ncols() {
        if i < feature_stats.feature_means.len() && i < feature_stats.feature_stds.len() {
            let col_mean = feature_stats.feature_means[i] as f64;
            let col_std = feature_stats.feature_stds[i] as f64;
            if col_std > 1e-8 {
                seq.column_mut(i).mapv_inplace(|x| (x - col_mean) / col_std);
            }
        }
    }
}

fn save_model_and_metrics(
    db_url: &str,
    model: &NeuralNetwork,
    asset: &str,
    model_type: ModelType,
    feature_stats: &crate::neural::data::FeatureStats,
    seq_length: usize,
    train_loss: f64,
    val_loss: f64,
    val_metrics: &crate::neural::model::ValidationMetrics,
    epoch: usize,
) -> Result<(), TrainingError> {
    
    debug!("💾 Salvando modelo e métricas");
    let mut save_client = connect_db(db_url)?;
    let mut weights = model.get_weights();
    
    // Configurar metadados do modelo
    weights.asset = asset.to_string();
    weights.model_type = model_type;
    weights.closing_mean = feature_stats.closing_mean as f64;
    weights.closing_std = feature_stats.closing_std as f64;
    weights.seq_length = seq_length;
    weights.epoch = epoch;
    weights.timestamp = Utc::now().to_rfc3339();

    let metrics = TrainingMetrics {
        asset: asset.to_string(),
        model_type: format!("{:?}", model_type),
        source: "database".to_string(),
        epoch,
        train_loss,
        val_loss,
        rmse: val_metrics.rmse,
        mae: val_metrics.mae,
        mape: val_metrics.mape,
        directional_accuracy: val_metrics.directional_accuracy,
        r_squared: val_metrics.r_squared,
        timestamp: Utc::now().to_rfc3339(),
    };

    save_model_to_postgres(&mut save_client, &weights, &metrics)?;
    debug!("✅ Modelo e métricas salvos com sucesso");
    Ok(())
}

// Exemplos de uso:
// cargo run --release -- --model-type lstm --asset PETR4 --seq-length 40 --epochs 100 --verbose
// cargo run --release -- --model-type rnn --all-assets --seq-length 30 --epochs 50 --batch-size 128 --use-scheduler
// cargo run --release -- --model-type mlp --asset VALE3 --hidden-size 256 --num-layers 4 --patience 20
// cargo run --release -- --model-type cnn --all-assets --dropout-rate 0.4 --l2-weight 0.001

// Example usage commands:
// cargo run --release -- --model-type lstm --asset ISAE4 --seq-length 30 --epochs 50 --verbose
// cargo run --release -- --model-type rnn --all-assets --seq-length 20 --epochs 30 --batch-size 64
// cargo run --release -- --model-type mlp --asset PETR4 --hidden-size 128 --epochs 100
// cargo run --release -- --model-type cnn --all-assets --seq-length 40 --num-layers 3