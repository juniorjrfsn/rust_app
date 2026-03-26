// projeto cnncheckin
// src/main.rs — CNN CheckIn — Reconhecimento de faces e objetos

mod camera;
mod cnn_model;
mod config;
mod database;
mod error;
mod face_detector;
mod image_processor;
mod gui;

use clap::{Parser, Subcommand};
use config::Config;
use database::Database;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

use cnn_model::{train_model, TrainingConfig};

// ────────────────────────────────────────────────
//  CLI
// ────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "cnncheckin",
    about = "Sistema de reconhecimento facial / objetos com CNN + KNN",
    version,
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Inicializa o sistema (banco de dados, diretórios)
    Init,

    /// Treina o modelo com as fotos do dataset (via CLI)
    Train {
        /// Diretório com as imagens de treino
        #[arg(short, long)]
        data_dir: Option<String>,

        /// Número de vizinhos KNN
        #[arg(short, long)]
        k: Option<usize>,

        /// Fator de augmentation de dados (padrão: 3)
        #[arg(short, long, default_value = "3")]
        augmentation: usize,

        /// Salvar modelo em arquivo JSON (além do banco)
        #[arg(long)]
        export: bool,
    },

    /// Comandos de banco de dados
    Database {
        #[command(subcommand)]
        action: DbCommands,
    },
}

#[derive(Subcommand)]
enum DbCommands {
    /// Cria as tabelas no banco
    Setup,

    /// Lista os modelos treinados
    Models,

    /// Lista as pessoas cadastradas
    Persons,

    /// Mostra estatísticas gerais
    Stats,

    /// Exporta um modelo para arquivo JSON
    Export {
        #[arg(short, long)]
        model_id: i32,
        #[arg(short, long, default_value = "model_export.json")]
        output: String,
    },

    /// Lista os check-ins recentes
    Checkins {
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
}

// ────────────────────────────────────────────────
//  main
// ────────────────────────────────────────────────

fn main() {
    // Inicializa logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    if let Some(cmd) = cli.command {
        if let Err(e) = run(cmd) {
            eprintln!("❌ Erro CLI: {}", e);
            std::process::exit(1);
        }
    } else {
        // Run GUI if no command is provided
        println!("🚀 Iniciando GUI...");
        let config = Config::load().expect("Erro ao carregar configuração");
        config.ensure_directories().expect("Erro ao criar diretórios");
        
        if let Err(e) = gui::run_gui(config) {
            eprintln!("❌ Erro na GUI: {}", e);
            std::process::exit(1);
        }
    }
}

fn run(cmd: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        // ── Init ────────────────────────────────
        Commands::Init => {
            println!("🚀 Inicializando CNN CheckIn CLI...");
            let config = Config::load()?;
            config.validate()?;
            config.ensure_directories()?;
            let _db = Database::new(&config.database)?;
            println!("✅ Sistema inicializado!");
            println!("   Banco: {:?}", config.database.path);
            println!("   Dataset: {}", config.paths.training_dir);
            println!("   Fotos: {}", config.paths.photos_dir);
        }

        // ── Train ───────────────────────────────
        Commands::Train { data_dir, k, augmentation, export } => {
            let config = Config::load()?;
            config.ensure_directories()?;

            let train_cfg = TrainingConfig {
                data_dir: data_dir.unwrap_or_else(|| config.paths.training_dir.clone()),
                epochs: config.model.epochs,
                k_neighbors: k.unwrap_or(config.model.k_neighbors),
                augmentation_factor: augmentation,
                validation_split: 0.2,
            };

            println!("🧠 Treinando modelo...");
            let start = Instant::now();

            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap(),
            );
            pb.set_message("Carregando e processando dataset...");
            pb.enable_steady_tick(std::time::Duration::from_millis(100));

            let model = train_model(&train_cfg)?;
            pb.finish_and_clear();

            let elapsed = start.elapsed();
            println!(
                "✅ Treinamento concluído em {:.1}s",
                elapsed.as_secs_f64()
            );
            println!("   Acurácia: {:.1}%", model.metadata.accuracy * 100.0);
            println!("   Classes: {}", model.metadata.num_classes);
            println!("   Classes: {:?}", model.metadata.class_names);

            // Salvar no banco
            let db = Database::new(&config.database)?;
            let model_id = db.save_model(&model)?;
            println!("   💾 Modelo salvo no banco com ID: {}", model_id);

            // Exportar para arquivo (opcional)
            if export {
                let filename = format!("{}/model_{}.json", config.paths.models_dir, model_id);
                model.save_to_file(&filename)?;
                println!("   📄 Modelo exportado: {}", filename);
            }
        }

        // ── Database ────────────────────────────
        Commands::Database { action } => {
            let config = Config::load()?;
            let db = Database::new(&config.database)?;

            match action {
                DbCommands::Setup => {
                    db.setup_tables()?;
                    println!("✅ Tabelas verificadas/criadas.");
                }

                DbCommands::Models => {
                    let models = db.list_models()?;
                    if models.is_empty() {
                        println!("Nenhum modelo treinado ainda.");
                    } else {
                        println!("{:<5} {:<30} {:<10} {:<8} {}", "ID", "Nome", "Acurácia", "Classes", "Criado em");
                        println!("{}", "─".repeat(80));
                        for m in models {
                            println!(
                                "{:<5} {:<30} {:>8.1}%  {:>6}   {}",
                                m.id.unwrap_or(0),
                                m.name,
                                m.accuracy * 100.0,
                                m.num_classes,
                                m.created_at
                            );
                        }
                    }
                }

                DbCommands::Persons => {
                    let persons = db.list_persons()?;
                    if persons.is_empty() {
                        println!("Nenhuma pessoa cadastrada ainda.");
                    } else {
                        println!("{:<5} {:<30} {:<8} {}", "ID", "Nome", "Fotos", "Cadastrado em");
                        println!("{}", "─".repeat(70));
                        for p in persons {
                            println!(
                                "{:<5} {:<30} {:>5}    {}",
                                p.id, p.name, p.photo_count, p.created_at
                            );
                        }
                    }
                }

                DbCommands::Stats => {
                    let stats = db.get_stats()?;
                    println!("📊 Estatísticas:");
                    println!("   Modelos treinados : {}", stats.total_models);
                    println!("   Pessoas cadastradas: {}", stats.total_persons);
                    println!("   Check-ins hoje    : {}", stats.checkins_today);
                }

                DbCommands::Export { model_id, output } => {
                    let model = db.load_model(model_id)?;
                    model.save_to_file(&output)?;
                    println!("✅ Modelo {} exportado para '{}'", model_id, output);
                }

                DbCommands::Checkins { limit } => {
                    let checkins = db.list_checkins(limit)?;
                    if checkins.is_empty() {
                        println!("Nenhum check-in registrado.");
                    } else {
                        println!("{:<5} {:<25} {:<10} {:<12} {}", "ID", "Pessoa", "Confiança", "Método", "Horário");
                        println!("{}", "─".repeat(80));
                        for c in checkins {
                            println!(
                                "{:<5} {:<25} {:>8.1}%  {:<12} {}",
                                c.id, c.person_name, c.confidence * 100.0, c.method, c.timestamp
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
