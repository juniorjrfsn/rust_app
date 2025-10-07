// file: cnncheckin/src/face_detector.rs
// Módulo de detecção e reconhecimento facial

use std::error::Error;
use std::time::{Duration, Instant};
use std::io::{self, Write};
use minifb::Key;
use ndarray::Array3;

use crate::camera::{WebcamCapture, capture_single_frame};
use crate::cnn_model::{TrainedModel, FaceRecognitionCNN, predict_face, create_face_embedding};
use crate::database::Database;
use crate::image_processor::{self, preprocess_image, detect_faces_in_image};
use crate::utils;

pub struct FaceDetector {
    model: Option<FaceRecognitionCNN<burn::backend::wgpu::WgpuBackend>>,
    database: Database,
    confidence_threshold: f32,
    similarity_threshold: f32,
}

impl FaceDetector {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let database = Database::new().await?;
        
        Ok(Self {
            model: None,
            database,
            confidence_threshold: 0.7,  // 70% confiança mínima
            similarity_threshold: 0.8,  // 80% similaridade mínima
        })
    }
    
    pub async fn load_model(&mut self, trained_model: TrainedModel) -> Result<(), Box<dyn Error>> {
        println!("🧠 Carregando modelo treinado...");
        
        let model = crate::cnn_model::load_model_for_inference(
            &trained_model.weights, 
            &trained_model.metadata
        ).await?;
        
        self.model = Some(model);
        
        println!("✅ Modelo carregado com sucesso!");
        println!("  📊 Classes: {}", trained_model.metadata.num_classes);
        println!("  🎯 Acurácia: {:.2}%", trained_model.metadata.accuracy * 100.0);
        
        Ok(())
    }
    
    pub async fn recognize_face(&self, image: &Array3<f32>) -> Result<RecognitionResult, Box<dyn Error>> {
        let model = self.model.as_ref()
            .ok_or("Modelo não carregado")?;
        
        // Predição com o modelo CNN
        let (class_id, confidence) = predict_face(model, image)?;
        
        if confidence < self.confidence_threshold {
            return Ok(RecognitionResult::Unknown(confidence));
        }
        
        // Criar embedding para comparação
        let embedding = create_face_embedding(model, image)?;
        
        // Buscar pessoa similar no banco
        let similar_person = self.database
            .find_similar_person(&embedding, self.similarity_threshold)
            .await?;
        
        match similar_person {
            Some(person) => {
                // Registrar check-in
                let checkin_id = self.database.record_checkin(
                    person.id,
                    confidence,
                    "recognition",
                    None,
                ).await?;
                
                Ok(RecognitionResult::Recognized {
                    person_name: person.name,
                    confidence,
                    person_id: person.id,
                    checkin_id,
                })
            }
            None => Ok(RecognitionResult::Unknown(confidence))
        }
    }
    
    pub async fn learn_face(&self, image: &Array3<f32>, person_name: &str) -> Result<i32, Box<dyn Error>> {
        let model = self.model.as_ref()
            .ok_or("Modelo não carregado")?;
        
        // Criar embedding
        let embedding = create_face_embedding(model, image)?;
        
        // Salvar pessoa no banco
        let person_id = self.database.save_person(person_name, &embedding).await?;
        
        // Registrar como aprendizado
        let _checkin_id = self.database.record_checkin(
            person_id,
            1.0, // Confiança máxima para aprendizado
            "learning",
            None,
        ).await?;
        
        println!("📚 Nova face aprendida: {} (ID: {})", person_name, person_id);
        
        Ok(person_id)
    }
}

#[derive(Debug)]
pub enum RecognitionResult {
    Recognized {
        person_name: String,
        confidence: f32,
        person_id: i32,
        checkin_id: i32,
    },
    Unknown(f32), // confidence score
}

pub async fn learning_mode(realtime: bool) -> Result<(), Box<dyn Error>> {
    println!("📚 Modo Aprendizado Ativado");
    
    let mut detector = FaceDetector::new().await?;
    
    // Tentar carregar modelo mais recente
    match detector.database.load_latest_model().await {
        Ok(model) => {
            detector.load_model(model).await?;
        }
        Err(_) => {
            println!("⚠️  Nenhum modelo encontrado. É necessário treinar primeiro.");
            return Ok(());
        }
    }
    
    if realtime {
        learning_mode_realtime(&detector).await
    } else {
        learning_mode_single_shot(&detector).await
    }
}

pub async fn recognition_mode(trained_model: TrainedModel, realtime: bool) -> Result<(), Box<dyn Error>> {
    println!("🔍 Modo Reconhecimento Ativado");
    
    let mut detector = FaceDetector::new().await?;
    detector.load_model(trained_model).await?;
    
    if realtime {
        recognition_mode_realtime(&detector).await
    } else {
        recognition_mode_single_shot(&detector).await
    }
}

async fn learning_mode_realtime(detector: &FaceDetector) -> Result<(), Box<dyn Error>> {
    let mut capture = WebcamCapture::new()?;
    
    println!("\n📹 Modo Aprendizado em Tempo Real");
    println!("📋 Controles:");
    println!("  ESPAÇO - Aprender face atual");
    println!("  ESC - Sair");
    println!("  R - Resetar contador FPS");
    
    let mut frame_count = 0u64;
    let mut last_fps_update = Instant::now();
    let mut learned_faces = 0;
    
    while capture.is_window_open() && !capture.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
        
        // Capturar frame
        if let Err(e) = capture.capture_frame() {
            eprintln!("❌ Erro ao capturar frame: {}", e);
            continue;
        }
        
        // Processar teclas
        if capture.is_key_pressed(Key::Space) {
            println!("📸 Capturando face para aprendizado...");
            
            // Obter frame atual
            let raw_frame = capture.get_current_frame();
            let (width, height) = capture.get_dimensions();
            
            // Detectar faces
            let faces = detect_faces_in_image(raw_frame, width as usize, height as usize)?;
            
            if faces.is_empty() {
                println!("❌ Nenhuma face detectada. Posicione-se melhor na câmera.");
                continue;
            }
            
            if faces.len() > 1 {
                println!("⚠️  Múltiplas faces detectadas. Mantenha apenas uma pessoa na tela.");
                continue;
            }
            
            // Preprocessar a primeira face detectada
            let face_image = preprocess_image(&faces[0])?;
            
            // Solicitar nome da pessoa
            print!("👤 Digite o nome da pessoa: ");
            io::stdout().flush()?;
            
            let mut person_name = String::new();
            io::stdin().read_line(&mut person_name)?;
            let person_name = person_name.trim();
            
            if !person_name.is_empty() {
                match detector.learn_face(&face_image, person_name).await {
                    Ok(person_id) => {
                        learned_faces += 1;
                        println!("✅ Face aprendida: {} (ID: {})", person_name, person_id);
                    }
                    Err(e) => eprintln!("❌ Erro ao aprender face: {}", e),
                }
            }
        }
        
        if capture.is_key_pressed(Key::R) {
            frame_count = 0;
            last_fps_update = Instant::now();
        }
        
        // Atualizar título
        frame_count += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_update) >= Duration::from_secs(1) {
            let fps = frame_count as f64 / now.duration_since(last_fps_update).as_secs_f64();
            frame_count = 0;
            last_fps_update = now;
            
            capture.update_title(&format!(
                "CNN CheckIn - Aprendizado - {:.1} FPS - {} faces aprendidas", 
                fps, learned_faces
            ));
        }
        
        // Manter taxa de quadros
        let frame_time = frame_start.elapsed();
        let target_frame_time = Duration::from_millis(33);
        if frame_time < target_frame_time {
            std::thread::sleep(target_frame_time - frame_time);
        }
    }
    
    println!("📊 Sessão de aprendizado finalizada: {} faces aprendidas", learned_faces);
    Ok(())
}

async fn learning_mode_single_shot(detector: &FaceDetector) -> Result<(), Box<dyn Error>> {
    println!("\n📷 Modo Aprendizado - Foto Única");
    println!("Pressione ENTER para capturar uma foto...");
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    // Capturar frame
    let raw_frame = capture_single_frame().await?;
    
    // Aqui assumimos dimensões padrão - em implementação real, pegar das configurações
    let faces = detect_faces_in_image(&raw_frame, 640, 480)?;
    
    if faces.is_empty() {
        println!("❌ Nenhuma face detectada na imagem.");
        return Ok(());
    }
    
    println!("✅ {} face(s) detectada(s)", faces.len());
    
    for (i, face) in faces.iter().enumerate() {
        let face_image = preprocess_image(face)?;
        
        print!("👤 Digite o nome da pessoa #{}: ", i + 1);
        io::stdout().flush()?;
        
        let mut person_name = String::new();
        io::stdin().read_line(&mut person_name)?;
        let person_name = person_name.trim();
        
        if !person_name.is_empty() {
            match detector.learn_face(&face_image, person_name).await {
                Ok(person_id) => {
                    println!("✅ Face aprendida: {} (ID: {})", person_name, person_id);
                }
                Err(e) => eprintln!("❌ Erro ao aprender face: {}", e),
            }
        }
    }
    
    Ok(())
}

async fn recognition_mode_realtime(detector: &FaceDetector) -> Result<(), Box<dyn Error>> {
    let mut capture = WebcamCapture::new()?;
    
    println!("\n👁️  Modo Reconhecimento em Tempo Real");
    println!("📋 Controles:");
    println!("  ESPAÇO - Reconhecer face atual");
    println!("  ESC - Sair");
    println!("  R - Resetar contador FPS");
    
    let mut frame_count = 0u64;
    let mut last_fps_update = Instant::now();
    let mut recognitions = 0;
    
    while capture.is_window_open() && !capture.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
        
        // Capturar frame
        if let Err(e) = capture.capture_frame() {
            eprintln!("❌ Erro ao capturar frame: {}", e);
            continue;
        }
        
        // Processar teclas
        if capture.is_key_pressed(Key::Space) {
            println!("🔍 Reconhecendo face...");
            
            let raw_frame = capture.get_current_frame();
            let (width, height) = capture.get_dimensions();
            
            let faces = detect_faces_in_image(raw_frame, width as usize, height as usize)?;
            
            if faces.is_empty() {
                println!("❌ Nenhuma face detectada.");
                continue;
            }
            
            for (i, face) in faces.iter().enumerate() {
                let face_image = preprocess_image(face)?;
                
                match detector.recognize_face(&face_image).await {
                    Ok(RecognitionResult::Recognized { 
                        person_name, 
                        confidence, 
                        person_id,
                        checkin_id 
                    }) => {
                        recognitions += 1;
                        println!("✅ Reconhecido: {} (Confiança: {:.2}%, ID: {}, Check-in: {})", 
                                person_name, confidence * 100.0, person_id, checkin_id);
                    }
                    Ok(RecognitionResult::Unknown(confidence)) => {
                        println!("❓ Pessoa desconhecida (Confiança: {:.2}%)", confidence * 100.0);
                    }
                    Err(e) => {
                        eprintln!("❌ Erro no reconhecimento: {}", e);
                    }
                }
            }
        }
        
        if capture.is_key_pressed(Key::R) {
            frame_count = 0;
            last_fps_update = Instant::now();
        }
        
        // Atualizar título
        frame_count += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_update) >= Duration::from_secs(1) {
            let fps = frame_count as f64 / now.duration_since(last_fps_update).as_secs_f64();
            frame_count = 0;
            last_fps_update = now;
            
            capture.update_title(&format!(
                "CNN CheckIn - Reconhecimento - {:.1} FPS - {} reconhecimentos", 
                fps, recognitions
            ));
        }
        
        // Manter taxa de quadros
        let frame_time = frame_start.elapsed();
        let target_frame_time = Duration::from_millis(33);
        if frame_time < target_frame_time {
            std::thread::sleep(target_frame_time - frame_time);
        }
    }
    
    println!("📊 Sessão de reconhecimento finalizada: {} reconhecimentos", recognitions);
    Ok(())
}

async fn recognition_mode_single_shot(detector: &FaceDetector) -> Result<(), Box<dyn Error>> {
    println!("\n👁️  Modo Reconhecimento - Foto Única");
    println!("Pressione ENTER para capturar e reconhecer...");
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let raw_frame = capture_single_frame().await?;
    let faces = detect_faces_in_image(&raw_frame, 640, 480)?;
    
    if faces.is_empty() {
        println!("❌ Nenhuma face detectada.");
        return Ok(());
    }
    
    println!("🔍 Analisando {} face(s)...", faces.len());
    
    for (i, face) in faces.iter().enumerate() {
        println!("\n--- Face {} ---", i + 1);
        
        let face_image = preprocess_image(face)?;
        
        match detector.recognize_face(&face_image).await {
            Ok(RecognitionResult::Recognized { 
                person_name, 
                confidence, 
                person_id,
                checkin_id 
            }) => {
                println!("✅ Reconhecido: {}", person_name);
                println!("   Confiança: {:.2}%", confidence * 100.0);
                println!("   ID da Pessoa: {}", person_id);
                println!("   Check-in ID: {}", checkin_id);
            }
            Ok(RecognitionResult::Unknown(confidence)) => {
                println!("❓ Pessoa desconhecida");
                println!("   Confiança: {:.2}%", confidence * 100.0);
            }
            Err(e) => {
                eprintln!("❌ Erro no reconhecimento da face {}: {}", i + 1, e);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[tokio::test]
    async fn test_face_detector_creation() {
        let detector = FaceDetector::new().await;
        assert!(detector.is_ok());
    }
    
    #[test]
    fn test_recognition_result() {
        let result = RecognitionResult::Unknown(0.5);
        match result {
            RecognitionResult::Unknown(conf) => assert_eq!(conf, 0.5),
            _ => panic!("Resultado inesperado"),
        }
    }
}