// file: cnncheckin/src/image_processor.rs
// Módulo de processamento de imagens e preparação de dados

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use ndarray::{Array3, Array4, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::utils;

#[derive(Debug, Clone)]
pub struct FaceImage {
    pub data: Array3<f32>, // [channels, height, width]
    pub person_name: String,
    pub class_id: usize,
    pub file_path: String,
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub confidence: f32,
    pub image_data: Array3<f32>,
}

pub struct FaceDataset {
    images: Vec<FaceImage>,
    class_names: Vec<String>,
    class_to_id: HashMap<String, usize>,
}

impl FaceDataset {
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            class_names: Vec::new(),
            class_to_id: HashMap::new(),
        }
    }

    pub fn add_image(&mut self, image: FaceImage) {
        // Adicionar nova classe se não existir
        if !self.class_to_id.contains_key(&image.person_name) {
            let class_id = self.class_names.len();
            self.class_names.push(image.person_name.clone());
            self.class_to_id.insert(image.person_name.clone(), class_id);
        }

        let mut image = image;
        image.class_id = self.class_to_id[&image.person_name];
        self.images.push(image);
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn get_class_names(&self) -> Vec<String> {
        self.class_names.clone()
    }

    pub fn split(&self, validation_ratio: f32) -> (FaceDataset, FaceDataset) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.images.len()).collect();
        indices.shuffle(&mut rng);

        let val_size = (self.images.len() as f32 * validation_ratio) as usize;
        let (val_indices, train_indices) = indices.split_at(val_size);

        let mut train_dataset = FaceDataset::new();
        let mut val_dataset = FaceDataset::new();

        // Copiar metadados
        train_dataset.class_names = self.class_names.clone();
        train_dataset.class_to_id = self.class_to_id.clone();
        val_dataset.class_names = self.class_names.clone();
        val_dataset.class_to_id = self.class_to_id.clone();

        // Dividir imagens
        for &idx in train_indices {
            train_dataset.images.push(self.images[idx].clone());
        }

        for &idx in val_indices {
            val_dataset.images.push(self.images[idx].clone());
        }

        println!("📊 Dataset dividido:");
        println!("  🏋️  Treino: {} imagens", train_dataset.len());
        println!("  ✅ Validação: {} imagens", val_dataset.len());

        (train_dataset, val_dataset)
    }

    pub fn get_batch(&self, batch_size: usize, start_idx: usize) -> Option<(Array4<f32>, Vec<usize>)> {
        let end_idx = std::cmp::min(start_idx + batch_size, self.images.len());
        
        if start_idx >= self.images.len() {
            return None;
        }

        let batch_images = &self.images[start_idx..end_idx];
        let actual_batch_size = batch_images.len();

        // Criar array 4D para as imagens [batch, channels, height, width]
        let mut images_array = Array4::<f32>::zeros((actual_batch_size, 3, 128, 128));
        let mut labels = Vec::new();

        for (i, face_image) in batch_images.iter().enumerate() {
            // Copiar dados da imagem para o batch
            images_array.slice_mut(s![i, .., .., ..]).assign(&face_image.data);
            labels.push(face_image.class_id);
        }

        Some((images_array, labels))
    }

    pub fn into_dataloader(self) -> DataLoader {
        DataLoader::new(self, 32)
    }
}

pub struct DataLoader {
    dataset: FaceDataset,
    batch_size: usize,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(dataset: FaceDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Array4<f32>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.dataset.get_batch(self.batch_size, self.current_idx);
        self.current_idx += self.batch_size;
        
        // Resetar se chegou ao fim
        if self.current_idx >= self.dataset.len() {
            self.current_idx = 0;
        }
        
        batch
    }
}

pub fn load_training_data(data_dir: &str) -> Result<FaceDataset, Box<dyn Error>> {
    println!("📁 Carregando dados de treinamento de: {}", data_dir);
    
    let mut dataset = FaceDataset::new();
    let data_path = Path::new(data_dir);
    
    if !data_path.exists() {
        return Err(format!("Diretório não encontrado: {}", data_dir).into());
    }
    
    // Procurar por subdiretórios (cada um representa uma pessoa)
    let person_dirs = fs::read_dir(data_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().ok().map_or(false, |ft| ft.is_dir()))
        .collect::<Vec<_>>();
    
    if person_dirs.is_empty() {
        return Err("Nenhum diretório de pessoa encontrado".into());
    }
    
    println!("👥 Encontrados {} diretórios de pessoas", person_dirs.len());
    
    for person_dir in person_dirs {
        let dir_path = person_dir.path();
        let dir_name = dir_path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        
        // Extrair nome da pessoa do nome do diretório (formato: "001_NomePessoa")
        let person_name = if let Some(underscore_pos) = dir_name.find('_') {
            &dir_name[underscore_pos + 1..]
        } else {
            dir_name
        };
        
        println!("📸 Processando pessoa: {}", person_name);
        
        // Carregar imagens do diretório
        let image_files = fs::read_dir(&dir_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().ok().map_or(false, |ft| ft.is_file()) &&
                entry.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map_or(false, |ext| matches!(ext.to_lowercase().as_str(), "ppm" | "jpg" | "jpeg" | "png"))
            })
            .collect::<Vec<_>>();
        
        println!("  📄 Encontradas {} imagens", image_files.len());
        
        for image_file in image_files {
            let file_path = image_file.path();
            
            match load_and_preprocess_image(&file_path) {
                Ok(processed_image) => {
                    let face_image = FaceImage {
                        data: processed_image,
                        person_name: person_name.to_string(),
                        class_id: 0, // Será atualizado pelo dataset
                        file_path: file_path.to_string_lossy().to_string(),
                    };
                    
                    dataset.add_image(face_image);
                }
                Err(e) => {
                    eprintln!("⚠️  Erro ao carregar {}: {}", file_path.display(), e);
                }
            }
        }
    }
    
    println!("✅ Dataset carregado:");
    println!("  📊 Total de imagens: {}", dataset.len());
    println!("  👥 Número de pessoas: {}", dataset.num_classes());
    println!("  📝 Pessoas: {:?}", dataset.get_class_names());
    
    if dataset.len() == 0 {
        return Err("Nenhuma imagem válida encontrada".into());
    }
    
    Ok(dataset)
}

fn load_and_preprocess_image(file_path: &Path) -> Result<Array3<f32>, Box<dyn Error>> {
    let extension = file_path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    let raw_image = match extension.as_str() {
        "ppm" => load_ppm_image(file_path)?,
        "jpg" | "jpeg" | "png" => {
            return Err("Suporte para JPG/PNG não implementado ainda. Use PPM.".into());
        }
        _ => {
            return Err(format!("Formato não suportado: {}", extension).into());
        }
    };
    
    // Preprocessar imagem
    preprocess_image(&raw_image)
}

fn load_ppm_image(file_path: &Path) -> Result<Array3<f32>, Box<dyn Error>> {
    let content = fs::read(file_path)?;
    
    // Parser simples para PPM P6
    let mut lines = content.split(|&b| b == b'\n');
    
    // Primeira linha: magic number
    let magic = lines.next().ok_or("PPM inválido: sem magic number")?;
    if magic != b"P6" {
        return Err("Apenas formato PPM P6 é suportado".into());
    }
    
    // Pular comentários
    let mut width_height_line = lines.next().ok_or("PPM inválido: sem dimensões")?;
    while width_height_line.starts_with(b"#") {
        width_height_line = lines.next().ok_or("PPM inválido: sem dimensões após comentários")?;
    }
    
    // Dimensões
    let dimensions = String::from_utf8(width_height_line.to_vec())?;
    let parts: Vec<&str> = dimensions.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return Err("PPM inválido: formato de dimensões incorreto".into());
    }
    
    let width: usize = parts[0].parse()?;
    let height: usize = parts[1].parse()?;
    
    // Valor máximo
    let max_val_line = lines.next().ok_or("PPM inválido: sem valor máximo")?;
    let max_val: u8 = String::from_utf8(max_val_line.to_vec())?.trim().parse()?;
    
    // Encontrar início dos dados binários
    let header_size = content.len() - lines.as_slice().len();
    let pixel_data = &content[header_size..];
    
    if pixel_data.len() < width * height * 3 {
        return Err("PPM inválido: dados de pixel insuficientes".into());
    }
    
    // Converter para Array3<f32> no formato [channels, height, width]
    let mut image = Array3::<f32>::zeros((3, height, width));
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            
            // Normalizar para [0, 1]
            let r = pixel_data[idx] as f32 / max_val as f32;
            let g = pixel_data[idx + 1] as f32 / max_val as f32;
            let b = pixel_data[idx + 2] as f32 / max_val as f32;
            
            image[[0, y, x]] = r; // Canal Red
            image[[1, y, x]] = g; // Canal Green
            image[[2, y, x]] = b; // Canal Blue
        }
    }
    
    Ok(image)
}

pub fn preprocess_image(raw_image: &Array3<f32>) -> Result<Array3<f32>, Box<dyn Error>> {
    let (channels, height, width) = raw_image.dim();
    
    if channels != 3 {
        return Err("Imagem deve ter 3 canais (RGB)".into());
    }
    
    // Redimensionar para 128x128 (tamanho esperado pelo modelo)
    let resized = resize_image(raw_image, 128, 128)?;
    
    // Normalização Z-score com valores típicos para imagens RGB
    let normalized = normalize_image(&resized)?;
    
    Ok(normalized)
}

fn resize_image(image: &Array3<f32>, target_width: usize, target_height: usize) -> Result<Array3<f32>, Box<dyn Error>> {
    let (channels, orig_height, orig_width) = image.dim();
    
    if orig_width == target_width && orig_height == target_height {
        return Ok(image.clone());
    }
    
    // Implementação simples de redimensionamento usando nearest neighbor
    let mut resized = Array3::<f32>::zeros((channels, target_height, target_width));
    
    let width_ratio = orig_width as f32 / target_width as f32;
    let height_ratio = orig_height as f32 / target_height as f32;
    
    for c in 0..channels {
        for y in 0..target_height {
            for x in 0..target_width {
                let orig_x = ((x as f32 + 0.5) * width_ratio - 0.5).max(0.0) as usize;
                let orig_y = ((y as f32 + 0.5) * height_ratio - 0.5).max(0.0) as usize;
                
                let orig_x = orig_x.min(orig_width - 1);
                let orig_y = orig_y.min(orig_height - 1);
                
                resized[[c, y, x]] = image[[c, orig_y, orig_x]];
            }
        }
    }
    
    Ok(resized)
}

fn normalize_image(image: &Array3<f32>) -> Result<Array3<f32>, Box<dyn Error>> {
    // Valores médios ImageNet para normalização
    let mean = [0.485, 0.456, 0.406]; // RGB
    let std = [0.229, 0.224, 0.225];  // RGB
    
    let mut normalized = image.clone();
    
    for c in 0..3 {
        let mut channel = normalized.slice_mut(s![c, .., ..]);
        channel -= mean[c];
        channel /= std[c];
    }
    
    Ok(normalized)
}

pub fn detect_faces_in_image(raw_frame: &[u8], width: usize, height: usize) -> Result<Vec<Array3<f32>>, Box<dyn Error>> {
    // Implementação simplificada de detecção de faces
    // Em uma implementação real, usaria OpenCV, MTCNN, ou outro detector
    
    if raw_frame.len() < width * height * 3 {
        return Err("Dados de frame insuficientes".into());
    }
    
    // Converter frame RGB para Array3
    let mut full_image = Array3::<f32>::zeros((3, height, width));
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            
            if idx + 2 < raw_frame.len() {
                full_image[[0, y, x]] = raw_frame[idx] as f32 / 255.0;     // R
                full_image[[1, y, x]] = raw_frame[idx + 1] as f32 / 255.0; // G
                full_image[[2, y, x]] = raw_frame[idx + 2] as f32 / 255.0; // B
            }
        }
    }
    
    // Detector simples: assume que toda a imagem é uma face
    // Em implementação real, usaria algoritmos de detecção mais sofisticados
    let detected_faces = vec![full_image];
    
    // Preprocessar cada face detectada
    let mut processed_faces = Vec::new();
    for face in detected_faces {
        match preprocess_image(&face) {
            Ok(processed) => processed_faces.push(processed),
            Err(e) => eprintln!("⚠️  Erro ao preprocessar face: {}", e),
        }
    }
    
    Ok(processed_faces)
}

pub fn count_training_images(data_dir: &str) -> Result<usize, Box<dyn Error>> {
    let data_path = Path::new(data_dir);
    
    if !data_path.exists() {
        return Ok(0);
    }
    
    let mut total_images = 0;
    
    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let person_dir = entry.path();
            
            let image_count = fs::read_dir(&person_dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.file_type().ok().map_or(false, |ft| ft.is_file()) &&
                    entry.path().extension()
                        .and_then(|ext| ext.to_str())
                        .map_or(false, |ext| matches!(ext.to_lowercase().as_str(), "ppm" | "jpg" | "jpeg" | "png"))
                })
                .count();
            
            total_images += image_count;
        }
    }
    
    Ok(total_images)
}

pub fn save_preprocessed_dataset(dataset: &FaceDataset, output_dir: &str) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    
    println!("💾 Salvando dataset preprocessado em: {}", output_dir);
    
    // Criar metadados do dataset
    let metadata = DatasetMetadata {
        num_images: dataset.len(),
        num_classes: dataset.num_classes(),
        class_names: dataset.get_class_names(),
        image_shape: [3, 128, 128],
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    // Salvar metadados
    let metadata_file = output_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(metadata_file, metadata_json)?;
    
    // Salvar imagens preprocessadas (formato binário)
    for (i, face_image) in dataset.images.iter().enumerate() {
        let filename = format!("image_{:06}_{}.bin", i, face_image.class_id);
        let file_path = output_path.join(filename);
        
        // Serializar Array3 usando bincode
        let serialized = bincode::serialize(&face_image.data.as_slice().unwrap())?;
        fs::write(file_path, serialized)?;
    }
    
    println!("✅ Dataset preprocessado salvo: {} imagens", dataset.len());
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct DatasetMetadata {
    num_images: usize,
    num_classes: usize,
    class_names: Vec<String>,
    image_shape: [usize; 3],
    created_at: String,
}

pub fn load_preprocessed_dataset(input_dir: &str) -> Result<FaceDataset, Box<dyn Error>> {
    let input_path = Path::new(input_dir);
    
    // Carregar metadados
    let metadata_file = input_path.join("metadata.json");
    let metadata_content = fs::read_to_string(metadata_file)?;
    let metadata: DatasetMetadata = serde_json::from_str(&metadata_content)?;
    
    println!("📁 Carregando dataset preprocessado:");
    println!("  📊 Imagens: {}", metadata.num_images);
    println!("  👥 Classes: {}", metadata.num_classes);
    
    let mut dataset = FaceDataset::new();
    
    // Recriar mapeamento de classes
    for (i, class_name) in metadata.class_names.iter().enumerate() {
        dataset.class_names.push(class_name.clone());
        dataset.class_to_id.insert(class_name.clone(), i);
    }
    
    // Carregar imagens
    for i in 0..metadata.num_images {
        for class_id in 0..metadata.num_classes {
            let filename = format!("image_{:06}_{}.bin", i, class_id);
            let file_path = input_path.join(&filename);
            
            if file_path.exists() {
                let serialized = fs::read(&file_path)?;
                let image_data: Vec<f32> = bincode::deserialize(&serialized)?;
                
                // Reconstruir Array3
                let image_array = Array3::from_shape_vec(
                    (metadata.image_shape[0], metadata.image_shape[1], metadata.image_shape[2]),
                    image_data
                )?;
                
                let face_image = FaceImage {
                    data: image_array,
                    person_name: metadata.class_names[class_id].clone(),
                    class_id,
                    file_path: file_path.to_string_lossy().to_string(),
                };
                
                dataset.images.push(face_image);
            }
        }
    }
    
    println!("✅ Dataset preprocessado carregado: {} imagens", dataset.len());
    Ok(dataset)
}

pub fn augment_dataset(dataset: &FaceDataset, augmentation_factor: usize) -> Result<FaceDataset, Box<dyn Error>> {
    println!("🔄 Aplicando data augmentation (fator: {}x)", augmentation_factor);
    
    let mut augmented_dataset = dataset.clone();
    let original_size = dataset.len();
    
    for face_image in &dataset.images {
        for _ in 0..(augmentation_factor - 1) { // -1 porque já temos a original
            let augmented_image = apply_random_augmentation(&face_image.data)?;
            
            let augmented_face = FaceImage {
                data: augmented_image,
                person_name: face_image.person_name.clone(),
                class_id: face_image.class_id,
                file_path: format!("{}_aug", face_image.file_path),
            };
            
            augmented_dataset.images.push(augmented_face);
        }
    }
    
    println!("✅ Dataset aumentado de {} para {} imagens", 
             original_size, augmented_dataset.len());
    
    Ok(augmented_dataset)
}

fn apply_random_augmentation(image: &Array3<f32>) -> Result<Array3<f32>, Box<dyn Error>> {
    let mut augmented = image.clone();
    
    // Aplicar algumas transformações aleatórias simples
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Flip horizontal (50% chance)
    if rng.gen_bool(0.5) {
        augmented = horizontal_flip(&augmented);
    }
    
    // Ajuste de brilho (+/- 10%)
    let brightness_factor = rng.gen_range(0.9..1.1);
    augmented *= brightness_factor;
    
    // Clamp valores para range válido
    augmented.mapv_inplace(|x| x.clamp(-2.0, 2.0)); // Considerando normalização ImageNet
    
    Ok(augmented)
}

fn horizontal_flip(image: &Array3<f32>) -> Array3<f32> {
    let (channels, height, width) = image.dim();
    let mut flipped = Array3::zeros((channels, height, width));
    
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                flipped[[c, y, x]] = image[[c, y, width - 1 - x]];
            }
        }
    }
    
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_face_dataset_creation() {
        let mut dataset = FaceDataset::new();
        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.num_classes(), 0);
        
        let dummy_image = Array3::<f32>::zeros((3, 128, 128));
        let face_image = FaceImage {
            data: dummy_image,
            person_name: "João".to_string(),
            class_id: 0,
            file_path: "test.ppm".to_string(),
        };
        
        dataset.add_image(face_image);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.num_classes(), 1);
    }
    
    #[test]
    fn test_image_normalization() {
        let image = Array3::<f32>::ones((3, 64, 64)) * 0.5; // Imagem cinza
        let normalized = normalize_image(&image);
        assert!(normalized.is_ok());
    }
    
    #[test]
    fn test_image_resize() {
        let image = Array3::<f32>::zeros((3, 64, 64));
        let resized = resize_image(&image, 128, 128);
        
        assert!(resized.is_ok());
        let resized = resized.unwrap();
        assert_eq!(resized.dim(), (3, 128, 128));
    }
    
    #[test]
    fn test_horizontal_flip() {
        let mut image = Array3::<f32>::zeros((3, 4, 4));
        image[[0, 0, 0]] = 1.0; // Marcar pixel superior esquerdo
        
        let flipped = horizontal_flip(&image);
        
        // Pixel deve estar no canto superior direito agora
        assert_eq!(flipped[[0, 0, 3]], 1.0);
        assert_eq!(flipped[[0, 0, 0]], 0.0);
    }
}