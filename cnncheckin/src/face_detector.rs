// projeto cnncheckin
// src/face_detector.rs — Detecção e reconhecimento facial

use ndarray::Array3;

use crate::cnn_model::{KnnClassifier, TrainedModel, load_model_for_inference};
use crate::config::Config;
use crate::database::Database;
use crate::error::{AppError, Result};

// ────────────────────────────────────────────────
//  Resultado de reconhecimento
// ────────────────────────────────────────────────

#[derive(Debug)]
pub enum RecognitionResult {
    Recognized {
        person_name: String,
        confidence: f32,
        person_id: i32,
        checkin_id: i32,
    },
    Unknown {
        confidence: f32,
    },
}

// ────────────────────────────────────────────────
//  FaceDetector
// ────────────────────────────────────────────────

pub struct FaceDetector {
    classifier: Option<KnnClassifier>,
    database: Database,
    confidence_threshold: f32,
    similarity_threshold: f32,
}

impl FaceDetector {
    pub fn new(config: &Config, database: Database) -> Result<Self> {
        Ok(Self {
            classifier: None,
            database,
            confidence_threshold: config.recognition.confidence_threshold,
            similarity_threshold: config.recognition.similarity_threshold,
        })
    }

    pub fn load_model(&mut self, trained_model: TrainedModel) -> &mut Self {
        println!(
            "🧠 Modelo carregado — {} classes | acurácia {:.1}%",
            trained_model.metadata.num_classes,
            trained_model.metadata.accuracy * 100.0
        );
        self.classifier = Some(load_model_for_inference(&trained_model));
        self
    }

    /// Reconhece uma face: busca no modelo e verifica embedding no banco
    pub fn recognize_face(&self, image: &Array3<f32>) -> Result<RecognitionResult> {
        let classifier = self.classifier.as_ref().ok_or(AppError::ModelNotTrained)?;

        let (class_id, confidence) = classifier.predict(image)?;

        if confidence < self.confidence_threshold {
            return Ok(RecognitionResult::Unknown { confidence });
        }

        // Gera embedding e busca pessoa similar no banco
        let embedding = classifier.embed(image);
        let similar = self.database.find_similar_person(&embedding, self.similarity_threshold)?;

        match similar {
            Some(person) => {
                let checkin_id = self.database.record_checkin(person.id, confidence, "recognition")?;
                Ok(RecognitionResult::Recognized {
                    person_name: person.name,
                    confidence,
                    person_id: person.id,
                    checkin_id,
                })
            }
            None => {
                // Confiança boa no modelo mas pessoa não cadastrada
                let class_name = classifier
                    .class_names
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| "desconhecido".into());
                println!("ℹ️  Classe {} reconhecida mas não encontrada no banco.", class_name);
                Ok(RecognitionResult::Unknown { confidence })
            }
        }
    }

    /// Aprende uma nova face: salva embedding no banco
    pub fn learn_face(&self, image: &Array3<f32>, person_name: &str) -> Result<i32> {
        let classifier = self.classifier.as_ref().ok_or(AppError::ModelNotTrained)?;
        let embedding = classifier.embed(image);
        let person_id = self.database.save_person(person_name, &embedding)?;
        self.database.record_checkin(person_id, 1.0, "learning")?;
        println!("📚 Face aprendida: {} (ID: {})", person_name, person_id);
        Ok(person_id)
    }
}