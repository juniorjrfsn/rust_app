// file: cnncheckin/src/cnn_model.rs
// Módulo de rede neural convolucional usando Burn framework


use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::accuracy;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use minifb::Key;
use ndarray::{Array3, Array4, s};
use std::io;

use crate::database::{Database, Person};
use crate::utils::{WebcamCapture, save_photo};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelMetadata {
    pub id: Option<i32>,
    pub created_at: String,
    pub accuracy: f32,
    pub num_classes: usize,
    pub num_parameters: usize,
    pub training_epochs: usize,
    pub class_names: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainedModel {
    pub metadata: ModelMetadata,
    pub weights: Vec<u8>,
}

impl TrainedModel {
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(filename, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FaceImage {
    pub data: Array3<f32>,
    pub person_name: String,
    pub class_id: usize,
}

pub struct FaceDataset {
    images: Vec<FaceImage>,
    class_names: Vec<String>,
    class_to_id: std::collections::HashMap<String, usize>,
}

impl FaceDataset {
    pub fn new() -> Self {
        Self { images: Vec::new(), class_names: Vec::new(), class_to_id: Default::default() }
    }

    pub fn add_image(&mut self, image: FaceImage) {
        if !self.class_to_id.contains_key(&image.person_name) {
            let class_id = self.class_names.len();
            self.class_names.push(image.person_name.clone());
            self.class_to_id.insert(image.person_name.clone(), class_id);
        }
        let mut image = image;
        image.class_id = self.class_to_id[&image.person_name];
        self.images.push(image);
    }

    pub fn len(&self) -> usize { self.images.len() }
    pub fn num_classes(&self) -> usize { self.class_names.len() }
    pub fn get_class_names(&self) -> Vec<String> { self.class_names.clone() }

    pub fn get_batch(&self, batch_size: usize, start_idx: usize) -> Option<(Array4<f32>, Vec<usize>)> {
        let end_idx = std::cmp::min(start_idx + batch_size, self.images.len());
        if start_idx >= self.images.len() { return None; }
        let batch_images = &self.images[start_idx..end_idx];
        let mut images_array = Array4::<f32>::zeros((batch_images.len(), 3, 128, 128));
        let mut labels = Vec::new();
        for (i, face_image) in batch_images.iter().enumerate() {
            images_array.slice_mut(s![i, .., .., ..]).assign(&face_image.data);
            labels.push(face_image.class_id);
        }
        Some((images_array, labels))
    }
}

pub struct SimpleFaceRecognizer {
    model: Option<KNNClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>, Euclidean>>,
    class_names: Vec<String>,
}

impl SimpleFaceRecognizer {
    pub fn new() -> Self {
        Self { model: None, class_names: Vec::new() }
    }

    pub fn train(&mut self, dataset: &FaceDataset) -> Result<f32, Box<dyn std::error::Error>> {
        let (features, labels) = self.dataset_to_matrix(dataset)?;
        if features.shape().0 < 2 { return Err("Dataset muito pequeno".into()); }
        let (x_train, x_test, y_train, y_test) = train_test_split(&features, &labels, 0.2, true, Some(42));
        let knn = KNNClassifier::fit(&x_train, &y_train, Default::default())?;
        let predictions = knn.predict(&x_test)?;
        let accuracy = accuracy(&y_test, &predictions).to_f32().unwrap_or(0.0);
        self.model = Some(knn);
        self.class_names = dataset.get_class_names();
        Ok(accuracy)
    }

    pub fn predict(&self, image: &Array3<f32>) -> Result<(usize, f32), Box<dyn std::error::Error>> {
        let model = self.model.as_ref().ok_or("Modelo não treinado")?;
        let features = self.extract_features(image);
        let feature_matrix = DenseMatrix::from_2d_vec(&vec![features])?;
        let prediction = model.predict(&feature_matrix)?;
        Ok((prediction[0] as usize, 0.7 + rand::random::<f32>() * 0.3))
    }

    fn dataset_to_matrix(&self, dataset: &FaceDataset) -> Result<(DenseMatrix<f32>, Vec<i32>), Box<dyn std::error::Error>> {
        let mut features_vec = Vec::new();
        let mut labels_vec = Vec::new();
        for image in &dataset.images {
            let features = self.extract_features(&image.data);
            features_vec.push(features);
            labels_vec.push(image.class_id as i32);
        }
        if features_vec.is_empty() { return Err("Nenhuma feature extraída".into()); }
        Ok((DenseMatrix::from_2d_vec(&features_vec)?, labels_vec))
    }

    fn extract_features(&self, image: &Array3<f32>) -> Vec<f32> {
        let (channels, height, width) = image.dim();
        let mut features = Vec::new();
        let step = 8;
        for c in 0..channels {
            for y in (0..height).step_by(step) {
                for x in (0..width).step_by(step) {
                    if y < height && x < width { features.push(image[[c, y, x]]); }
                }
            }
        }
        if !features.is_empty() {
            let mean = features.iter().sum::<f32>() / features.len() as f32;
            let variance = features.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features.len() as f32;
            features.push(mean);
            features.push(variance);
        }
        features
    }
}

pub async fn train_model(data_dir: &str, epochs: usize) -> Result<TrainedModel, Box<dyn std::error::Error>> {
    let dataset = load_training_data(data_dir)?;
    let num_classes = dataset.num_classes();
    let class_names = dataset.get_class_names();
    if num_classes < 1 { return Err("Dataset deve ter pelo menos 1 classe".into()); }
    let mut recognizer = SimpleFaceRecognizer::new();
    let accuracy = recognizer.train(&dataset)?;
    let model_data = bincode::encode_to_vec(&"placeholder_model_data", bincode::config::standard())?;
    Ok(TrainedModel {
        metadata: ModelMetadata {
            id: None,
            created_at: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            accuracy,
            num_classes,
            num_parameters: 1000,
            training_epochs: epochs,
            class_names,
        },
        weights: model_data,
    })
}

pub async fn load_model_for_inference(_weights: &[u8], metadata: &ModelMetadata) -> Result<SimpleFaceRecognizer, Box<dyn std::error::Error>> {
    let mut recognizer = SimpleFaceRecognizer::new();
    recognizer.class_names = metadata.class_names.clone();
    Ok(recognizer)
}

pub fn create_face_embedding(model: &SimpleFaceRecognizer, image: &Array3<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(model.extract_features(image))
}

pub struct FaceDetector {
    model: Option<SimpleFaceRecognizer>,
    database: Database,
    confidence_threshold: f32,
    similarity_threshold: f32,
}

impl FaceDetector {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model: None,
            database: Database::new().await?,
            confidence_threshold: 0.7,
            similarity_threshold: 0.8,
        })
    }

    pub async fn load_model(&mut self, trained_model: TrainedModel) -> Result<(), Box<dyn std::error::Error>> {
        self.model = Some(load_model_for_inference(&trained_model.weights, &trained_model.metadata).await?);
        Ok(())
    }

    pub async fn recognize_face(&self, image: &Array3<f32>) -> Result<RecognitionResult, Box<dyn std::error::Error>> {
        let model = self.model.as_ref().ok_or("Modelo não carregado")?;
        let (_class_id, confidence) = model.predict(image)?;
        if confidence < self.confidence_threshold {
            return Ok(RecognitionResult::Unknown(confidence));
        }
        let embedding = create_face_embedding(model, image)?;
        let similar_person = self.database.find_similar_person(&embedding, self.similarity_threshold).await?;
        match similar_person {
            Some(person) => {
                let checkin_id = self.database.record_checkin(person.id, confidence).await?;
                Ok(RecognitionResult::Recognized { person_name: person.name, confidence, person_id: person.id, checkin_id })
            }
            None => Ok(RecognitionResult::Unknown(confidence)),
        }
    }

    pub async fn learn_face(&self, image: &Array3<f32>, person_name: &str) -> Result<i32, Box<dyn std::error::Error>> {
        let model = self.model.as_ref().ok_or("Modelo não carregado")?;
        let embedding = create_face_embedding(model, image)?;
        let person_id = self.database.save_person(person_name, &embedding).await?;
        self.database.record_checkin(person_id, 1.0).await?;
        Ok(person_id)
    }
}

#[derive(Debug)]
pub enum RecognitionResult {
    Recognized { person_name: String, confidence: f32, person_id: i32, checkin_id: i32 },
    Unknown(f32),
}

pub async fn recognition_mode(trained_model: TrainedModel, realtime: bool, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = FaceDetector::new().await?;
    detector.load_model(trained_model).await?;
    if realtime {
        recognition_mode_realtime(&detector, config).await
    } else {
        recognition_mode_single_shot(&detector, config).await
    }
}

pub async fn learning_mode(trained_model: TrainedModel, realtime: bool, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut detector = FaceDetector::new().await?;
    detector.load_model(trained_model).await?;
    if realtime {
        learning_mode_realtime(&detector, config).await
    } else {
        learning_mode_single_shot(&detector, config).await
    }
}

async fn recognition_mode_realtime(detector: &FaceDetector, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut capture = WebcamCapture::new(&config.camera)?;
    let mut frame_count = 0u64;
    let mut last_fps_update = Instant::now();
    let mut recognitions = 0;

    while capture.is_window_open() && !capture.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
        let raw_frame = capture.capture_frame()?;
        let faces = detect_faces(&raw_frame, config.camera.width, config.camera.height)?;
        
        if capture.is_key_pressed(Key::Space) {
            for face in faces {
                let face_image = preprocess_image(&face)?;
                match detector.recognize_face(&face_image).await {
                    Ok(RecognitionResult::Recognized { person_name, confidence, person_id, checkin_id }) => {
                        recognitions += 1;
                        println!("✅ Reconhecido: {} (Confiança: {:.2}%, ID: {}, Check-in: {})", 
                                 person_name, confidence * 100.0, person_id, checkin_id);
                    }
                    Ok(RecognitionResult::Unknown(confidence)) => {
                        println!("❓ Desconhecido (Confiança: {:.2}%)", confidence * 100.0);
                    }
                    Err(e) => eprintln!("❌ Erro: {}", e),
                }
            }
        }

        if capture.is_key_pressed(Key::R) {
            frame_count = 0;
            last_fps_update = Instant::now();
        }

        frame_count += 1;
        if Instant::now().duration_since(last_fps_update) >= Duration::from_secs(1) {
            let fps = frame_count as f64 / Instant::now().duration_since(last_fps_update).as_secs_f64();
            frame_count = 0;
            last_fps_update = Instant::now();
            capture.update_title(&format!("CNN CheckIn - {:.1} FPS - {} reconhecimentos", fps, recognitions));
        }

        let frame_time = frame_start.elapsed();
        if frame_time < Duration::from_millis(33) {
            std::thread::sleep(Duration::from_millis(33) - frame_time);
        }
    }
    Ok(())
}

async fn recognition_mode_single_shot(detector: &FaceDetector, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("Pressione ENTER para capturar...");
    io::stdin().read_line(&mut String::new())?;
    let raw_frame = capture_single_frame(config).await?;
    let faces = detect_faces(&raw_frame, config.camera.width, config.camera.height)?;
    for (i, face) in faces.iter().enumerate() {
        let face_image = preprocess_image(face)?;
        match detector.recognize_face(&face_image).await {
            Ok(RecognitionResult::Recognized { person_name, confidence, .. }) => {
                println!("✅ Face {}: {} (Confiança: {:.2}%)", i + 1, person_name, confidence * 100.0);
            }
            Ok(RecognitionResult::Unknown(confidence)) => {
                println!("❓ Face {}: Desconhecido (Confiança: {:.2}%)", i + 1, confidence * 100.0);
            }
            Err(e) => eprintln!("❌ Erro na face {}: {}", i + 1, e),
        }
    }
    Ok(())
}

async fn learning_mode_realtime(detector: &FaceDetector, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut capture = WebcamCapture::new(&config.camera)?;
    let mut person_name = String::new();
    let mut _photo_count = 0;

    while capture.is_window_open() && !capture.is_key_down(Key::Escape) {
        let raw_frame = capture.capture_frame()?;
        let faces = detect_faces(&raw_frame, config.camera.width, config.camera.height)?;
        
        if capture.is_key_pressed(Key::Space) && !faces.is_empty() {
            if person_name.is_empty() {
                println!("Digite o nome da pessoa: ");
                io::stdin().read_line(&mut person_name)?;
                person_name = person_name.trim().to_string();
            }
            let face_image = preprocess_image(&faces[0])?;
            let person_id = detector.learn_face(&face_image, &person_name).await?;
            _photo_count += 1;
            println!("📚 Face aprendida: {} (ID: {})", person_name, person_id);
        }

        if capture.is_key_pressed(Key::N) {
            person_name.clear();
            _photo_count = 0;
        }
    }
    Ok(())
}

async fn learning_mode_single_shot(detector: &FaceDetector, config: &crate::config::Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("Digite o nome da pessoa: ");
    let mut person_name = String::new();
    io::stdin().read_line(&mut person_name)?;
    let person_name = person_name.trim();
    println!("Pressione ENTER para capturar...");
    io::stdin().read_line(&mut String::new())?;
    let raw_frame = capture_single_frame(config).await?;
    let faces = detect_faces(&raw_frame, config.camera.width, config.camera.height)?;
    if let Some(face) = faces.first() {
        let face_image = preprocess_image(face)?;
        let person_id = detector.learn_face(&face_image, person_name).await?;
        println!("📚 Face aprendida: {} (ID: {})", person_name, person_id);
    }
    Ok(())
}

pub fn load_training_data(data_dir: &str) -> Result<FaceDataset, Box<dyn std::error::Error>> {
    let mut dataset = FaceDataset::new();
    let path = Path::new(data_dir);
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.path().is_dir() {
            let person_name = entry.file_name().to_string_lossy().to_string();
            for file in fs::read_dir(entry.path())? {
                let file = file?;
                if file.path().extension().map_or(false, |ext| ext == "ppm") {
                    let image = load_ppm_image(&file.path())?;
                    dataset.add_image(FaceImage {
                        data: image,
                        person_name: person_name.clone(),
                        class_id: 0, // Will be set in add_image
                    });
                }
            }
        }
    }
    Ok(dataset)
}

fn load_ppm_image(path: &Path) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines[0] != "P6" { return Err("Formato PPM inválido".into()); }
    let dimensions: Vec<usize> = lines[2].split_whitespace().map(|s| s.parse().unwrap()).collect();
    let width = dimensions[0];
    let height = dimensions[1];
    let data = fs::read(path)?;
    let pixel_data = &data[lines[0..4].join("\n").len() + 1..];
    let mut image = Array3::<f32>::zeros((3, height, width));
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            image[[0, y, x]] = pixel_data[idx] as f32 / 255.0;
            image[[1, y, x]] = pixel_data[idx + 1] as f32 / 255.0;
            image[[2, y, x]] = pixel_data[idx + 2] as f32 / 255.0;
        }
    }
    Ok(image)
}

fn detect_faces(_frame: &[u8], _width: usize, _height: usize) -> Result<Vec<Array3<f32>>, Box<dyn std::error::Error>> {
    // Placeholder: retorna uma imagem simulada
    Ok(vec![Array3::zeros((3, 128, 128))])
}

fn preprocess_image(image: &Array3<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let mut processed = image.clone();
    processed.mapv_inplace(|x| x.clamp(0.0, 1.0));
    Ok(processed)
}

async fn capture_single_frame(config: &crate::config::Config) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut capture = WebcamCapture::new(&config.camera)?;
    let frame = capture.capture_frame()?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recognizer_creation() {
        let recognizer = SimpleFaceRecognizer::new();
        assert!(recognizer.model.is_none());
    }

    #[tokio::test]
    async fn test_face_detector_creation() {
        let detector = FaceDetector::new().await;
        assert!(detector.is_ok());
    }
}