// projeto cnncheckin
// src/gui.rs — Implementação do eframe (egui) interface do usuário

use eframe::egui;
use std::sync::mpsc::{channel, Receiver};
use std::thread;

use crate::camera::{WebcamCapture, sanitize_name};
use crate::config::Config;
use crate::database::Database;
use crate::face_detector::{FaceDetector, RecognitionResult};
use crate::image_processor::detect_faces;
use crate::cnn_model::{train_model, TrainingConfig};

pub fn run_gui(config: Config) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("CNN CheckIn - Reconhecimento Facial"),
        ..Default::default()
    };
    
    eframe::run_native(
        "CNN CheckIn",
        options,
        Box::new(|cc| Box::new(App::new(config, cc))),
    )
}

#[derive(PartialEq)]
enum AppTab {
    Recognition,
    Training,
}

struct App {
    config: Config,
    tab: AppTab,
    capture: WebcamCapture,
    detector: Option<FaceDetector>,
    database: Database,
    
    // Training state
    train_person_name: String,
    train_photos_taken: u32,
    train_target_count: u32,
    train_status_text: String,
    
    // Background threading
    is_training: bool,
    train_receiver: Option<Receiver<Result<String, String>>>,
    
    // Recognition state
    last_recognition_text: String,
}

impl App {
    fn new(config: Config, _cc: &eframe::CreationContext<'_>) -> Self {
        let database = Database::new(&config.database).expect("Falha ao abrir o banco de dados.");
        let capture = WebcamCapture::new(&config.camera).expect("Falha ao inicializar câmera");
        
        // Carrega o modelo se existir
        let mut detector = None;
        if let Ok(model) = database.load_latest_model() {
            let mut d = FaceDetector::new(&config, database.clone()).expect("Falha ao criar FaceDetector");
            d.load_model(model);
            detector = Some(d);
        }

        Self {
            config,
            tab: AppTab::Recognition,
            capture,
            detector,
            database,
            train_person_name: String::new(),
            train_photos_taken: 0,
            train_target_count: 10,
            train_status_text: String::new(),
            is_training: false,
            train_receiver: None,
            last_recognition_text: "Aguardando...".to_string(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Verifica se o treinamento em background terminou
        if self.is_training {
            if let Some(rx) = &self.train_receiver {
                if let Ok(result) = rx.try_recv() {
                    self.is_training = false;
                    match result {
                        Ok(msg) => {
                            self.train_status_text = msg;
                            // Recarrega o novo modelo no detector
                            if let Ok(model) = self.database.load_latest_model() {
                                let mut d = FaceDetector::new(&self.config, self.database.clone()).unwrap();
                                d.load_model(model);
                                self.detector = Some(d);
                            }
                        }
                        Err(e) => {
                            self.train_status_text = format!("Erro ao treinar: {}", e);
                        }
                    }
                }
            }
        }

        // Atualiza a câmera
        let _ = self.capture.capture_frame();

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, AppTab::Recognition, "Reconhecimento Facial");
                ui.selectable_value(&mut self.tab, AppTab::Training, "Treinamento");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let frame = self.capture.current_frame_data();
            let (w, h) = self.capture.dimensions();
            
            // Renderiza imagem da câmera
            let color_image = egui::ColorImage::from_rgb([w, h], frame);
            let texture = ctx.load_texture("camera_frame", color_image, Default::default());
            ui.image(&texture);

            ui.separator();

            match self.tab {
                AppTab::Recognition => {
                    ui.heading("Reconhecimento");
                    ui.label(&self.last_recognition_text);
                    
                    if let Some(detector) = &self.detector {
                        let faces = detect_faces(frame, w, h);
                        if let Some(face) = faces.first() {
                            match detector.recognize_face(face) {
                                Ok(RecognitionResult::Recognized { person_name, confidence, .. }) => {
                                    self.last_recognition_text = format!("Reconhecido: {} ({:.1}%)", person_name, confidence * 100.0);
                                }
                                Ok(RecognitionResult::Unknown { confidence }) => {
                                    self.last_recognition_text = format!("Desconhecido ({:.1}%)", confidence * 100.0);
                                }
                                Err(e) => {
                                    self.last_recognition_text = format!("Erro: {}", e);
                                }
                            }
                        } else {
                            self.last_recognition_text = "Nenhuma face detectada.".to_string();
                        }
                    } else {
                        self.last_recognition_text = "Modelo não carregado. Vá para a guia de Treinamento.".to_string();
                    }
                }
                AppTab::Training => {
                    ui.heading("Capturar Fotos & Treinamento");
                    ui.horizontal(|ui| {
                        ui.label("Nome da Pessoa:");
                        ui.text_edit_singleline(&mut self.train_person_name);
                    });
                    
                    ui.label(format!("Fotos capturadas: {} / {}", self.train_photos_taken, self.train_target_count));
                    
                    ui.add_enabled_ui(!self.is_training, |ui| {
                        if ui.button("Capturar Foto").clicked() && !self.train_person_name.is_empty() {
                            let save_dir = format!("{}/{}", self.config.paths.training_dir, sanitize_name(&self.train_person_name));
                            std::fs::create_dir_all(&save_dir).ok();
                            let filename = format!("{}/photo_{:04}.ppm", save_dir, self.train_photos_taken + 1);
                            let path = std::path::Path::new(&filename);
                            
                            if let Ok(_) = self.capture.save_current_frame(path) {
                                self.train_photos_taken += 1;
                                self.train_status_text = format!("Foto salva: {}/{}", self.train_photos_taken, self.train_target_count);
                            } else {
                                self.train_status_text = "Erro ao salvar foto.".to_string();
                            }
                        }
                    });
                    
                    ui.separator();
                    ui.label(&self.train_status_text);
                    
                    ui.add_enabled_ui(!self.is_training, |ui| {
                        if ui.button("Treinar Modelo Agora").clicked() {
                            self.is_training = true;
                            self.train_status_text = "Treinando o modelo em background (isso pode demorar uns instantes)...".to_string();
                            
                            let (tx, rx) = channel();
                            self.train_receiver = Some(rx);
                            
                            let train_cfg = TrainingConfig {
                                data_dir: self.config.paths.training_dir.clone(),
                                epochs: self.config.model.epochs,
                                k_neighbors: self.config.model.k_neighbors,
                                augmentation_factor: 3,
                                validation_split: 0.2,
                            };
                            
                            let db_clone = self.database.clone();
                            
                            thread::spawn(move || {
                                match train_model(&train_cfg) {
                                    Ok(model) => {
                                        match db_clone.save_model(&model) {
                                            Ok(id) => {
                                                let msg = format!("Modelo salvo com sucesso no banco com ID {}!", id);
                                                let _ = tx.send(Ok(msg));
                                            }
                                            Err(e) => {
                                                let _ = tx.send(Err(format!("Erro ao slavar no banco: {}", e)));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(e.to_string()));
                                    }
                                }
                            });
                        }
                    });
                }
            }
        });

        // Request repaint continuously to simulate video feed and keep checking thread
        ctx.request_repaint();
    }
}
