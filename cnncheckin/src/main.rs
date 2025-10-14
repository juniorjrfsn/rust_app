// projeto: cnncheckin
// file: cnncheckin/src/main.rs
// Sistema modular de reconhecimento facial com CNN
 

mod camera;
mod config;
mod database;
mod error;
mod image;
mod model;
mod recognition;
mod training;
mod utils;

use clap::{Parser, Subcommand};
use error::Result;

#[derive(Parser)]
#[command(name = "cnncheckin")]
#[command(about = "Sistema de reconhecimento facial com CNN", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inicializar configuração e banco de dados
    Init,
    
    /// Capturar imagens da webcam para treinamento
    Capture {
        /// Número de fotos por pessoa
        #[arg(short, long, default_value = "10")]
        count: u32,
        
        /// Nome da pessoa (opcional, será solicitado durante captura se não fornecido)
        #[arg(short, long)]
        person: Option<String>,
    },
    
    /// Treinar modelo de reconhecimento facial
    Train {
        /// Diretório com imagens de treino
        #[arg(short, long)]
        data_dir: Option<String>,
        
        /// Número de épocas
        #[arg(short, long)]
        epochs: Option<usize>,
        
        /// Salvar modelo automaticamente após treino
        #[arg(long)]
        auto_save: bool,
    },
    
    /// Reconhecer faces
    Recognize {
        /// Modo tempo real (webcam)
        #[arg(short, long)]
        realtime: bool,
        
        /// ID do modelo a usar (usa o mais recente se não especificado)
        #[arg(short, long)]
        model_id: Option<i32>,
    },
    
    /// Aprender novas faces
    Learn {
        /// Modo tempo real (webcam)
        #[arg(short, long)]
        realtime: bool,
    },
    
    /// Gerenciar modelos
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },
    
    /// Gerenciar banco de dados
    Database {
        #[command(subcommand)]
        action: DatabaseCommands,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// Listar modelos salvos
    List,
    
    /// Exportar modelo
    Export {
        #[arg(short, long)]
        model_id: i32,
        
        #[arg(short, long)]
        output: String,
    },
    
    /// Importar modelo
    Import {
        #[arg(short, long)]
        input: String,
    },
    
    /// Deletar modelo
    Delete {
        #[arg(short, long)]
        model_id: i32,
    },
    
    /// Comparar modelos
    Compare,
}

#[derive(Subcommand)]
enum DatabaseCommands {
    /// Configurar tabelas
    Setup,
    
    /// Estatísticas
    Stats,
    
    /// Backup
    Backup {
        #[arg(short, long)]
        output: String,
    },
    
    /// Restaurar backup
    Restore {
        #[arg(short, long)]
        input: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Inicializar logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Init => {
            commands::init().await?;
        }
        
        Commands::Capture { count, person } => {
            commands::capture(count, person).await?;
        }
        
        Commands::Train { data_dir, epochs, auto_save } => {
            commands::train(data_dir, epochs, auto_save).await?;
        }
        
        Commands::Recognize { realtime, model_id } => {
            commands::recognize(realtime, model_id).await?;
        }
        
        Commands::Learn { realtime } => {
            commands::learn(realtime).await?;
        }
        
        Commands::Model { action } => {
            commands::model(action).await?;
        }
        
        Commands::Database { action } => {
            commands::database(action).await?;
        }
    }
    
    Ok(())
}

mod commands {
    use super::*;
    use crate::config::Config;
    use crate::database::Database;
    
    pub async fn init() -> Result<()> {
        println!("🚀 Inicializando CNN CheckIn...");
        
        // Criar configuração padrão
        let config = Config::load_or_create()?;
        config.ensure_directories()?;
        config.validate()?;
        
        println!("✅ Configuração criada/carregada");
        
        // Configurar banco de dados
        let db = Database::connect().await?;
        db.setup_tables().await?;
        
        println!("✅ Banco de dados configurado");
        println!("\n📋 Sistema pronto para uso!");
        println!("💡 Próximos passos:");
        println!("   1. Capture imagens: cnncheckin capture");
        println!("   2. Treine o modelo: cnncheckin train");
        println!("   3. Reconheça faces: cnncheckin recognize --realtime");
        
        Ok(())
    }
    
    pub async fn capture(count: u32, person: Option<String>) -> Result<()> {
        let config = Config::load()?;
        let mut session = training::CaptureSession::new(config).await?;
        session.capture_dataset(count, person).await?;
        Ok(())
    }
    
    pub async fn train(
        data_dir: Option<String>,
        epochs: Option<usize>,
        auto_save: bool,
    ) -> Result<()> {
        let config = Config::load()?;
        let data_dir = data_dir.unwrap_or_else(|| config.paths.training_dir.clone());
        
        let mut trainer = training::ModelTrainer::new(config)?;
        
        if let Some(epochs) = epochs {
            trainer.set_epochs(epochs);
        }
        
        let trained_model = trainer.train(&data_dir).await?;
        
        if auto_save {
            let db = Database::connect().await?;
            let model_id = db.save_model(&trained_model).await?;
            println!("✅ Modelo salvo com ID: {}", model_id);
        }
        
        Ok(())
    }
    
    pub async fn recognize(realtime: bool, model_id: Option<i32>) -> Result<()> {
        let config = Config::load()?;
        let db = Database::connect().await?;
        
        let model = if let Some(id) = model_id {
            db.load_model(id).await?
        } else {
            db.load_latest_model().await?
        };
        
        let mut recognizer = recognition::FaceRecognizer::new(config, db).await?;
        recognizer.load_model(model).await?;
        
        if realtime {
            recognizer.recognize_realtime().await?;
        } else {
            recognizer.recognize_single_shot().await?;
        }
        
        Ok(())
    }
    
    pub async fn learn(realtime: bool) -> Result<()> {
        let config = Config::load()?;
        let db = Database::connect().await?;
        
        let model = db.load_latest_model().await?;
        
        let mut recognizer = recognition::FaceRecognizer::new(config, db).await?;
        recognizer.load_model(model).await?;
        
        if realtime {
            recognizer.learn_realtime().await?;
        } else {
            recognizer.learn_single_shot().await?;
        }
        
        Ok(())
    }
    
    pub async fn model(action: ModelCommands) -> Result<()> {
        use crate::model::ModelManager;
        
        let db = Database::connect().await?;
        let manager = ModelManager::new(db);
        
        match action {
            ModelCommands::List => manager.list_models().await?,
            ModelCommands::Export { model_id, output } => {
                manager.export_model(model_id, &output).await?
            }
            ModelCommands::Import { input } => manager.import_model(&input).await?,
            ModelCommands::Delete { model_id } => manager.delete_model(model_id).await?,
            ModelCommands::Compare => manager.compare_models().await?,
        }
        
        Ok(())
    }
    
    pub async fn database(action: DatabaseCommands) -> Result<()> {
        let db = Database::connect().await?;
        
        match action {
            DatabaseCommands::Setup => {
                db.setup_tables().await?;
                println!("✅ Tabelas configuradas");
            }
            DatabaseCommands::Stats => {
                db.print_statistics().await?;
            }
            DatabaseCommands::Backup { output } => {
                db.backup(&output).await?;
            }
            DatabaseCommands::Restore { input } => {
                db.restore(&input).await?;
            }
        }
        
        Ok(())
    }
}

// Next Steps After Compilation
// Once it compiles successfully:

// Test basic functionality: cargo run -- database setup
// Test image capture: cargo run -- capture --count 5
