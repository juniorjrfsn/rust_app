// projeto: cnncheckin
// file: cnncheckin/src/main.rs
// Sistema modular de reconhecimento facial com CNN
 
 
 // projeto: cnncheckin
// file: cnncheckin/src/main.rs
// Sistema modular de reconhecimento facial com CNN

mod camera;
mod cnn_model;
mod database;
mod face_detector;
mod image_processor;
mod config;
mod utils;

use clap::{Parser, Subcommand};
use std::error::Error;
use std::io;

#[derive(Parser)]
#[command(name = "cnncheckin")]
#[command(about = "Sistema de reconhecimento facial com CNN")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Capturar imagens da webcam
    Capture {
        /// Número de fotos para treino por pessoa
        #[arg(short, long, default_value = "10")]
        count: u32,
    },
    /// Treinar modelo CNN
    Train {
        /// Diretório com imagens de treino
        #[arg(short, long, default_value = "../../dados/fotos_treino")]
        input_dir: String,
    },
    /// Reconhecer faces
    Recognize {
        /// Usar webcam em tempo real
        #[arg(short, long)]
        realtime: bool,
    },
    /// Gerenciar banco de dados
    Database {
        #[command(subcommand)]
        action: DatabaseCommands,
    },
}

#[derive(Subcommand)]
enum DatabaseCommands {
    /// Criar tabelas
    Setup,
    /// Listar modelos salvos
    List,
    /// Exportar modelo
    Export {
        #[arg(short, long)]
        model_id: i32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    
    let cli = Cli::parse();

    match cli.command {
        Commands::Capture { count } => {
            println!("Modo Captura de Imagens");
            println!("Pressione ENTER para comecar a capturar {} fotos por pessoa", count);
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            camera::capture_training_images(count).await?;
        },
        
        Commands::Train { input_dir } => {
            println!("Modo Treinamento CNN");
            
            // Verificar se há imagens suficientes
            let image_count = image_processor::count_training_images(&input_dir)?;
            if image_count < 20 {
                println!("Aviso: Encontradas apenas {} imagens. Recomendado: pelo menos 20", image_count);
                println!("Deseja continuar? (s/N): ");
                
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                
                if !input.trim().to_lowercase().starts_with('s') {
                    return Ok(());
                }
            }
            
            let model = cnn_model::train_model(&input_dir).await?;
            
            // Salvar modelo no banco de dados
            println!("Salvando modelo no banco de dados...");
            let db = database::Database::new().await?;
            let model_id = db.save_model(&model).await?;
            
            println!("Modelo salvo com ID: {}", model_id);
        },
        
        Commands::Recognize { realtime } => {
            println!("Modo Reconhecimento Facial");
            
            // Perguntar que tipo de reconhecimento
            println!("Selecione o tipo de reconhecimento:");
            println!("1) Aprendizado (adicionar novas faces)");
            println!("2) Reconhecimento (identificar faces conhecidas)");
            print!("Opcao (1-2): ");
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "1" => {
                    println!("Modo Aprendizado ativado");
                    face_detector::learning_mode(realtime).await?;
                },
                "2" => {
                    println!("Modo Reconhecimento ativado");
                    
                    // Carregar modelo do banco
                    let db = database::Database::new().await?;
                    let latest_model = db.load_latest_model().await?;
                    
                    face_detector::recognition_mode(latest_model, realtime).await?;
                },
                _ => {
                    println!("Opcao invalida");
                    return Ok(());
                }
            }
        },
        
        Commands::Database { action } => {
            let db = database::Database::new().await?;
            
            match action {
                DatabaseCommands::Setup => {
                    println!("Configurando banco de dados...");
                    db.setup_tables().await?;
                    println!("Tabelas criadas com sucesso!");
                },
                
                DatabaseCommands::List => {
                    println!("Modelos salvos:");
                    let models = db.list_models().await?;
                    for model in models {
                        println!("ID: {:?}, Data: {}, Acuracia: {:.2}%", 
                                model.id, model.created_at, model.accuracy * 100.0);
                    }
                },
                
                DatabaseCommands::Export { model_id } => {
                    println!("Exportando modelo ID: {}", model_id);
                    let model = db.load_model(model_id).await?;
                    
                    let filename = format!("modelo_{}.json", model_id);
                    model.save_to_file(&filename)?;
                    println!("Modelo exportado para: {}", filename);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_database_connection() {
        let db_result = database::Database::new().await;
        assert!(db_result.is_ok(), "Falha na conexao com banco de dados");
    }
    
    #[test]
    fn test_config_load() {
        let config = config::Config::load();
        assert!(config.is_ok(), "Falha ao carregar configuracao");
    }
}

// Next Steps After Compilation
// Once it compiles successfully:

// Test basic functionality: cargo run -- database setup
// Test image capture: cargo run -- capture --count 5
