// projeto cnncheckin
// file: cnncheckin/src/cnn_model.rs
// Módulo de rede neural convolucional usando Burn framework
// src/cnn_model.rs — Modelo de reconhecimento usando KNN com features CNN-like

use ndarray::Array3;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::error::{AppError, Result};
use crate::image_processor::{extract_features, FaceDataset, load_training_data, augment_dataset};

// ────────────────────────────────────────────────
//  Metadados e modelo serializado
// ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: Option<i32>,
    pub name: String,
    pub created_at: String,
    pub accuracy: f32,
    pub num_classes: usize,
    pub training_epochs: usize,
    pub class_names: Vec<String>,
    pub k_neighbors: usize,
}

/// Modelo treinado pronto para serialização / armazenamento
#[derive(Serialize, Deserialize)]
pub struct TrainedModel {
    pub metadata: ModelMetadata,
    /// Protótipos de treino: (features, class_id)
    pub prototypes: Vec<(Vec<f32>, usize)>,
}

impl TrainedModel {
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| AppError::Generic(e.to_string()))?;
        fs::write(path, json).map_err(AppError::Io)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let json = fs::read_to_string(path).map_err(AppError::Io)?;
        serde_json::from_str(&json).map_err(|e| AppError::Generic(e.to_string()))
    }
}

// ────────────────────────────────────────────────
//  Classificador KNN
// ────────────────────────────────────────────────

/// Classificador KNN em memória pura (sem dependências externas de ML)
pub struct KnnClassifier {
    /// (features_vetor, class_id)
    prototypes: Vec<(Vec<f32>, usize)>,
    pub class_names: Vec<String>,
    k: usize,
}

impl KnnClassifier {
    pub fn new(k: usize) -> Self {
        Self {
            prototypes: Vec::new(),
            class_names: Vec::new(),
            k,
        }
    }

    /// Treina o classificador com o dataset inteiro
    pub fn fit(&mut self, dataset: &FaceDataset) -> Result<f32> {
        if dataset.len() < 2 {
            return Err(AppError::InsufficientData);
        }

        self.class_names = dataset.get_class_names();
        self.prototypes.clear();

        let (features, labels) = dataset.to_feature_matrix();
        for (feat, label) in features.iter().zip(labels.iter()) {
            self.prototypes.push((feat.clone(), *label));
        }

        // Avalia com leave-one-out (rápido para datasets pequenos)
        let accuracy = self.evaluate_loo()?;
        Ok(accuracy)
    }

    /// Leave-one-out cross-validation
    fn evaluate_loo(&self) -> Result<f32> {
        if self.prototypes.len() < 2 {
            return Ok(0.0);
        }

        let mut correct = 0usize;
        let total = self.prototypes.len();

        for i in 0..total {
            let (test_feat, true_label) = &self.prototypes[i];

            // KNN excluindo o próprio ponto
            let mut distances: Vec<(f32, usize)> = self
                .prototypes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (feat, label))| (euclidean_distance(test_feat, feat), *label))
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k = self.k.min(distances.len());
            let predicted = majority_vote(&distances[..k]);

            if predicted == *true_label {
                correct += 1;
            }
        }

        Ok(correct as f32 / total as f32)
    }

    /// Prediz a classe de uma imagem e retorna (class_id, confidence)
    pub fn predict(&self, image: &Array3<f32>) -> Result<(usize, f32)> {
        if self.prototypes.is_empty() {
            return Err(AppError::ModelNotTrained);
        }

        let features = extract_features(image);
        let mut distances: Vec<(f32, usize)> = self
            .prototypes
            .iter()
            .map(|(feat, label)| (euclidean_distance(&features, feat), *label))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let k = self.k.min(distances.len());
        let neighbors = &distances[..k];

        let predicted_class = majority_vote(neighbors);

        // Confiança = fração dos k vizinhos que pertencem à classe predita
        let votes_for_class = neighbors
            .iter()
            .filter(|(_, label)| *label == predicted_class)
            .count();
        let confidence = votes_for_class as f32 / k as f32;

        Ok((predicted_class, confidence))
    }

    /// Retorna o embedding (features) de uma imagem
    pub fn embed(&self, image: &Array3<f32>) -> Vec<f32> {
        extract_features(image)
    }

    /// Exporta os protótipos para serialização
    pub fn export_prototypes(&self) -> Vec<(Vec<f32>, usize)> {
        self.prototypes.clone()
    }

    /// Importa protótipos de um modelo salvo
    pub fn import_prototypes(&mut self, prototypes: Vec<(Vec<f32>, usize)>, class_names: Vec<String>, k: usize) {
        self.prototypes = prototypes;
        self.class_names = class_names;
        self.k = k;
    }

    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn is_trained(&self) -> bool {
        !self.prototypes.is_empty()
    }
}

// ────────────────────────────────────────────────
//  Funções auxiliares de distância
// ────────────────────────────────────────────────

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn majority_vote(neighbors: &[(f32, usize)]) -> usize {
    let mut votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (_, label) in neighbors {
        *votes.entry(*label).or_insert(0) += 1;
    }
    *votes.iter().max_by_key(|(_, &v)| v).map(|(k, _)| k).unwrap_or(&0)
}

// ────────────────────────────────────────────────
//  Pipeline de treinamento
// ────────────────────────────────────────────────

pub struct TrainingConfig {
    pub data_dir: String,
    pub epochs: usize,
    pub k_neighbors: usize,
    pub augmentation_factor: usize,
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data_dir: "dados/fotos_treino".into(),
            epochs: 1, // KNN não tem epochs reais; usado para augmentation loops
            k_neighbors: 5,
            augmentation_factor: 3,
            validation_split: 0.2,
        }
    }
}

/// Treina o modelo completo e retorna um TrainedModel
pub fn train_model(cfg: &TrainingConfig) -> Result<TrainedModel> {
    println!("🚀 Iniciando treinamento...");

    // 1) Carregar dados
    let dataset = load_training_data(&cfg.data_dir)?;

    if dataset.num_classes() < 1 {
        return Err(AppError::InsufficientData);
    }

    println!(
        "📊 Dataset: {} imagens / {} classes",
        dataset.len(),
        dataset.num_classes()
    );

    // 2) Augmentation
    let augmented = if cfg.augmentation_factor > 1 {
        augment_dataset(&dataset, cfg.augmentation_factor)
    } else {
        dataset.clone()
    };

    // 3) Split treino / validação
    let (train_ds, val_ds) = augmented.split(cfg.validation_split);

    // 4) Treinar KNN
    let mut classifier = KnnClassifier::new(cfg.k_neighbors);
    println!("🧠 Treinando KNN (k={})...", cfg.k_neighbors);
    let train_accuracy = classifier.fit(&train_ds)?;
    println!("  ✅ Acurácia treino (LOO): {:.1}%", train_accuracy * 100.0);

    // 5) Avaliar no conjunto de validação
    let val_accuracy = evaluate(&classifier, &val_ds);
    println!("  ✅ Acurácia validação: {:.1}%", val_accuracy * 100.0);

    // 6) Montar modelo treinado
    let metadata = ModelMetadata {
        id: None,
        name: format!("model_{}", chrono::Utc::now().timestamp()),
        created_at: chrono::Utc::now().to_rfc3339(),
        accuracy: val_accuracy,
        num_classes: classifier.num_classes(),
        training_epochs: cfg.epochs,
        class_names: classifier.class_names.clone(),
        k_neighbors: cfg.k_neighbors,
    };

    Ok(TrainedModel {
        metadata,
        prototypes: classifier.export_prototypes(),
    })
}

/// Avalia o classificador em um dataset de validação
pub fn evaluate(classifier: &KnnClassifier, dataset: &FaceDataset) -> f32 {
    if dataset.is_empty() {
        return 0.0;
    }

    let mut correct = 0usize;
    for face_img in &dataset.images {
        if let Ok((predicted, _)) = classifier.predict(&face_img.data) {
            if predicted == face_img.class_id {
                correct += 1;
            }
        }
    }

    correct as f32 / dataset.len() as f32
}

/// Carrega um modelo salvo e reconstrói o KnnClassifier
pub fn load_model_for_inference(model: &TrainedModel) -> KnnClassifier {
    let mut classifier = KnnClassifier::new(model.metadata.k_neighbors);
    classifier.import_prototypes(
        model.prototypes.clone(),
        model.metadata.class_names.clone(),
        model.metadata.k_neighbors,
    );
    classifier
}

// ────────────────────────────────────────────────
//  Testes
// ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_processor::FaceImage;
    use ndarray::Array3;

    fn make_dataset(n_per_class: usize) -> FaceDataset {
        let mut ds = FaceDataset::new();
        for class in ["Alice", "Bob"] {
            for i in 0..n_per_class {
                let mut img = Array3::<f32>::zeros((3, 128, 128));
                // Imagens da classe têm um padrão distinto
                if class == "Alice" {
                    img[[0, 0, 0]] = 1.0;
                } else {
                    img[[2, 0, 0]] = 1.0;
                }
                ds.add_image(FaceImage {
                    data: img,
                    person_name: class.into(),
                    class_id: 0,
                    file_path: format!("{}_{}.ppm", class, i),
                });
            }
        }
        ds
    }

    #[test]
    fn test_knn_fit_predict() {
        let ds = make_dataset(5);
        let mut knn = KnnClassifier::new(3);
        let acc = knn.fit(&ds).unwrap();
        assert!(acc > 0.0);

        let test_img = Array3::<f32>::zeros((3, 128, 128));
        let (class_id, confidence) = knn.predict(&test_img).unwrap();
        assert!(class_id < 2);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_knn_untrained_error() {
        let knn = KnnClassifier::new(3);
        let img = Array3::<f32>::zeros((3, 128, 128));
        let result = knn.predict(&img);
        assert!(matches!(result, Err(AppError::ModelNotTrained)));
    }
}