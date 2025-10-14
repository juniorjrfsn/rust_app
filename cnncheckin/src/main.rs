



mod config;
mod database;
mod cnn_model;
mod utils;

use clap::{Parser, Subcommand};
use config::Config;
use database::Database;
use cnn_model::{train_model, recognition_mode, learning_mode};
use utils::WebcamCapture;

#[derive(Parser)]
#[command(name = "cnncheckin", about = "Sistema de reconhecimento facial com CNN", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Capture { #[arg(short, long, default_value = "10")] count: u32, #[arg(short, long)] person: Option<String> },
    Train { #[arg(short, long)] data_dir: Option<String>, #[arg(short, long)] epochs: Option<usize> },
    Recognize { #[arg(short, long)] realtime: bool, #[arg(short, long)] model_id: Option<i32> },
    Learn { #[arg(short, long)] realtime: bool },
    Database { #[command(subcommand)] action: DatabaseCommands },
}

#[derive(Subcommand)]
enum DatabaseCommands {
    Setup,
    List,
    Stats,
    Export { #[arg(short, long)] model_id: i32 },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            let config = Config::load()?;
            config.ensure_directories()?;
            config.validate()?;
            let db = Database::new().await?;
            db.setup_tables().await?;
            println!("✅ Sistema inicializado!");
            Ok(())
        }
        Commands::Capture { count, person } => {
            let config = Config::load()?;
            let mut capture = WebcamCapture::new(&config.camera)?;
            capture.capture_dataset(count, person).await?;
            Ok(())
        }
        Commands::Train { data_dir, epochs } => {
            let config = Config::load()?;
            let data_dir = data_dir.unwrap_or(config.paths.training_dir);
            let model = train_model(&data_dir, epochs.unwrap_or(config.model.epochs)).await?;
            let db = Database::new().await?;
            let model_id = db.save_model(&model).await?;
            println!("✅ Modelo salvo com ID: {}", model_id);
            Ok(())
        }
        Commands::Recognize { realtime, model_id } => {
            let config = Config::load()?;
            let db = Database::new().await?;
            let model = match model_id {
                Some(id) => db.load_model(id).await?,
                None => db.load_latest_model().await?,
            };
            recognition_mode(model, realtime, &config).await?;
            Ok(())
        }
        Commands::Learn { realtime } => {
            let config = Config::load()?;
            let db = Database::new().await?;
            let model = db.load_latest_model().await?;
            learning_mode(model, realtime, &config).await?;
            Ok(())
        }
        Commands::Database { action } => {
            let db = Database::new().await?;
            match action {
                DatabaseCommands::Setup => {
                    db.setup_tables().await?;
                    println!("✅ Tabelas configuradas");
                }
                DatabaseCommands::List => {
                    let models = db.list_models().await?;
                    for model in models {
                        println!("ID: {}, Acurácia: {:.2}%, Classes: {}", model.id.unwrap(), model.accuracy * 100.0, model.num_classes);
                    }
                }
                DatabaseCommands::Stats => {
                    let stats = db.get_stats().await?;
                    println!("📊 Estatísticas: {} modelos, {} pessoas, {} check-ins hoje", stats.models, stats.persons, stats.checkins_today);
                }
                DatabaseCommands::Export { model_id } => {
                    let model = db.load_model(model_id).await?;
                    model.save_to_file(&format!("model_{}.json", model_id))?;
                    println!("✅ Modelo exportado");
                }
            }
            Ok(())
        }
    }
}

// Next Steps After Compilation
// Once it compiles successfully:

// Test basic functionality: cargo run -- database setup
// Test image capture: cargo run -- capture --count 5
