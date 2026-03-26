// projeto cnncheckin
// file: cnncheckin/src/image_processor.rs
// Módulo de processamento de imagens e preparação de dados
// src/image_processor.rs — Processamento de imagens e gerenciamento do dataset

use ndarray::{Array3, Array4, s};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{AppError, Result};

// ────────────────────────────────────────────────
//  Estruturas de dados
// ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FaceImage {
    pub data: Array3<f32>,    // [channels, height, width], valores em [0, 1]
    pub person_name: String,
    pub class_id: usize,
    pub file_path: String,
}

/// Dataset de imagens organizadas por classe
pub struct FaceDataset {
    pub images: Vec<FaceImage>,
    pub class_names: Vec<String>,
    class_to_id: HashMap<String, usize>,
}

impl Clone for FaceDataset {
    fn clone(&self) -> Self {
        FaceDataset {
            images: self.images.clone(),
            class_names: self.class_names.clone(),
            class_to_id: self.class_to_id.clone(),
        }
    }
}

impl FaceDataset {
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            class_names: Vec::new(),
            class_to_id: HashMap::new(),
        }
    }

    pub fn add_image(&mut self, mut image: FaceImage) {
        if !self.class_to_id.contains_key(&image.person_name) {
            let class_id = self.class_names.len();
            self.class_names.push(image.person_name.clone());
            self.class_to_id.insert(image.person_name.clone(), class_id);
        }
        image.class_id = self.class_to_id[&image.person_name];
        self.images.push(image);
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn get_class_names(&self) -> Vec<String> {
        self.class_names.clone()
    }

    /// Separa o dataset em treino e validação de forma estratificada
    pub fn split(&self, validation_ratio: f32) -> (FaceDataset, FaceDataset) {
        let mut rng = rand::thread_rng();
        let mut train_ds = FaceDataset::new();
        let mut val_ds = FaceDataset::new();

        // Copiar mapeamento de classes para ambos
        train_ds.class_names = self.class_names.clone();
        train_ds.class_to_id = self.class_to_id.clone();
        val_ds.class_names = self.class_names.clone();
        val_ds.class_to_id = self.class_to_id.clone();

        // Estratificar por classe
        for class_id in 0..self.class_names.len() {
            let mut class_images: Vec<&FaceImage> = self
                .images
                .iter()
                .filter(|img| img.class_id == class_id)
                .collect();
            class_images.shuffle(&mut rng);

            let val_size = ((class_images.len() as f32) * validation_ratio).ceil() as usize;
            let val_size = val_size.max(1).min(class_images.len());

            for (i, img) in class_images.iter().enumerate() {
                if i < val_size {
                    val_ds.images.push((*img).clone());
                } else {
                    train_ds.images.push((*img).clone());
                }
            }
        }

        println!(
            "📊 Dataset dividido — Treino: {} | Validação: {}",
            train_ds.len(),
            val_ds.len()
        );

        (train_ds, val_ds)
    }

    /// Retorna um batch como (Array4[batch, C, H, W], labels)
    pub fn get_batch(&self, batch_size: usize, start_idx: usize) -> Option<(Array4<f32>, Vec<usize>)> {
        if start_idx >= self.images.len() {
            return None;
        }
        let end_idx = (start_idx + batch_size).min(self.images.len());
        let batch = &self.images[start_idx..end_idx];

        let n = batch.len();
        let (_, h, w) = batch[0].data.dim();
        let c = batch[0].data.shape()[0];

        let mut images_array = Array4::<f32>::zeros((n, c, h, w));
        let mut labels = Vec::with_capacity(n);

        for (i, face_img) in batch.iter().enumerate() {
            images_array.slice_mut(s![i, .., .., ..]).assign(&face_img.data);
            labels.push(face_img.class_id);
        }

        Some((images_array, labels))
    }

    /// Retorna features achatadas (para KNN): Vec<Vec<f32>> e Vec<usize>
    pub fn to_feature_matrix(&self) -> (Vec<Vec<f32>>, Vec<usize>) {
        let features: Vec<Vec<f32>> = self
            .images
            .iter()
            .map(|img| extract_features(&img.data))
            .collect();
        let labels: Vec<usize> = self.images.iter().map(|img| img.class_id).collect();
        (features, labels)
    }
}

// ────────────────────────────────────────────────
//  Carregamento de dados
// ────────────────────────────────────────────────

/// Carrega dataset de uma pasta organizada em subpastas por pessoa/objeto
pub fn load_training_data(data_dir: &str) -> Result<FaceDataset> {
    let path = Path::new(data_dir);
    if !path.exists() {
        return Err(AppError::Config(format!(
            "Diretório não encontrado: {}",
            data_dir
        )));
    }

    let mut dataset = FaceDataset::new();

    let person_dirs: Vec<_> = fs::read_dir(path)
        .map_err(AppError::Io)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map_or(false, |t| t.is_dir()))
        .collect();

    if person_dirs.is_empty() {
        return Err(AppError::InsufficientData);
    }

    println!("👥 Encontradas {} classes", person_dirs.len());

    for dir_entry in &person_dirs {
        let dir_path = dir_entry.path();
        let raw_name = dir_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Aceita tanto "001_Nome" quanto "Nome" como nome do diretório
        let person_name = if let Some(pos) = raw_name.find('_') {
            if raw_name[..pos].chars().all(|c| c.is_ascii_digit()) {
                raw_name[pos + 1..].to_string()
            } else {
                raw_name.to_string()
            }
        } else {
            raw_name.to_string()
        };

        let image_files: Vec<_> = fs::read_dir(&dir_path)
            .map_err(AppError::Io)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().ok().map_or(false, |t| t.is_file())
                    && is_supported_image(e.path().extension().and_then(|x| x.to_str()).unwrap_or(""))
            })
            .collect();

        println!("  📸 {}: {} imagens", person_name, image_files.len());

        for file_entry in image_files {
            let file_path = file_entry.path();
            match load_and_preprocess_image(&file_path, 128, 128) {
                Ok(data) => {
                    dataset.add_image(FaceImage {
                        data,
                        person_name: person_name.clone(),
                        class_id: 0, // corrigido em add_image
                        file_path: file_path.to_string_lossy().into_owned(),
                    });
                }
                Err(e) => {
                    eprintln!("⚠️  Ignorando {}: {}", file_path.display(), e);
                }
            }
        }
    }

    if dataset.is_empty() {
        return Err(AppError::InsufficientData);
    }

    println!(
        "✅ Dataset: {} imagens | {} classes",
        dataset.len(),
        dataset.num_classes()
    );

    Ok(dataset)
}

fn is_supported_image(ext: &str) -> bool {
    matches!(ext.to_lowercase().as_str(), "ppm" | "pgm" | "png" | "jpg" | "jpeg")
}

// ────────────────────────────────────────────────
//  Carregamento e pré-processamento de imagem
// ────────────────────────────────────────────────

/// Carrega uma imagem de disco e redimensiona para (C, height, width)
pub fn load_and_preprocess_image(path: &Path, target_h: usize, target_w: usize) -> Result<Array3<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let raw = match ext.as_str() {
        "ppm" => load_ppm(path)?,
        "pgm" => load_pgm(path)?,
        "png" | "jpg" | "jpeg" => load_pnm_fallback(path)?,
        _ => return Err(AppError::Image(format!("Formato não suportado: {}", ext))),
    };

    let resized = resize_image(&raw, target_h, target_w);
    Ok(normalize_image(&resized))
}

/// Carrega PPM P6 (binário RGB)
fn load_ppm(path: &Path) -> Result<Array3<f32>> {
    let bytes = fs::read(path).map_err(AppError::Io)?;
    let mut pos = 0usize;

    let magic = read_token(&bytes, &mut pos);
    if magic != "P6" {
        return Err(AppError::Image(format!("Não é PPM P6: {}", path.display())));
    }

    skip_whitespace_comments(&bytes, &mut pos);
    let width: usize = read_token(&bytes, &mut pos)
        .parse()
        .map_err(|_| AppError::Image("Largura inválida no PPM".into()))?;
    skip_whitespace_comments(&bytes, &mut pos);
    let height: usize = read_token(&bytes, &mut pos)
        .parse()
        .map_err(|_| AppError::Image("Altura inválida no PPM".into()))?;
    skip_whitespace_comments(&bytes, &mut pos);
    let _maxval: usize = read_token(&bytes, &mut pos)
        .parse()
        .map_err(|_| AppError::Image("Maxval inválido no PPM".into()))?;
    // Pular exatamente 1 byte de espaço em branco após maxval
    pos += 1;

    let expected = width * height * 3;
    if bytes.len() - pos < expected {
        return Err(AppError::Image(format!(
            "PPM corrompido: esperado {} bytes, encontrei {}",
            expected,
            bytes.len() - pos
        )));
    }

    let pixel_data = &bytes[pos..pos + expected];
    let mut image = Array3::<f32>::zeros((3, height, width));
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            image[[0, y, x]] = pixel_data[i] as f32;
            image[[1, y, x]] = pixel_data[i + 1] as f32;
            image[[2, y, x]] = pixel_data[i + 2] as f32;
        }
    }
    Ok(image) // Normalização acontece depois
}

/// Carrega PGM P5 (binário tons de cinza) e replica para 3 canais
fn load_pgm(path: &Path) -> Result<Array3<f32>> {
    let bytes = fs::read(path).map_err(AppError::Io)?;
    let mut pos = 0usize;

    let magic = read_token(&bytes, &mut pos);
    if magic != "P5" {
        return Err(AppError::Image("Não é PGM P5".into()));
    }

    skip_whitespace_comments(&bytes, &mut pos);
    let width: usize = read_token(&bytes, &mut pos).parse().unwrap_or(0);
    skip_whitespace_comments(&bytes, &mut pos);
    let height: usize = read_token(&bytes, &mut pos).parse().unwrap_or(0);
    skip_whitespace_comments(&bytes, &mut pos);
    let _maxval: usize = read_token(&bytes, &mut pos).parse().unwrap_or(255);
    pos += 1;

    let pixel_data = &bytes[pos..];
    let mut image = Array3::<f32>::zeros((3, height, width));
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let v = *pixel_data.get(i).unwrap_or(&0) as f32;
            image[[0, y, x]] = v;
            image[[1, y, x]] = v;
            image[[2, y, x]] = v;
        }
    }
    Ok(image)
}

/// Fallback: tenta ler como PPM P3 (ASCII) para PNG/JPEG (sem biblioteca nativa)
/// Em produção, adicione a crate `image` para suporte real a PNG/JPEG.
fn load_pnm_fallback(path: &Path) -> Result<Array3<f32>> {
    // Tenta como PPM P6 primeiro
    load_ppm(path).or_else(|_| {
        Err(AppError::Image(format!(
            "Para carregar PNG/JPEG, adicione a crate `image` ao Cargo.toml. Arquivo: {}",
            path.display()
        )))
    })
}

// ── Helpers de parsing de PNM ───────────────────

fn read_token(bytes: &[u8], pos: &mut usize) -> String {
    skip_whitespace_comments(bytes, pos);
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
    String::from_utf8_lossy(&bytes[start..*pos]).into_owned()
}

fn skip_whitespace_comments(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() {
        if bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
        } else if bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        } else {
            break;
        }
    }
}

// ────────────────────────────────────────────────
//  Transformações de imagem
// ────────────────────────────────────────────────

/// Redimensiona uma imagem [C, H, W] para [C, new_h, new_w] usando nearest-neighbor
pub fn resize_image(image: &Array3<f32>, new_h: usize, new_w: usize) -> Array3<f32> {
    let (channels, orig_h, orig_w) = image.dim();
    if orig_h == new_h && orig_w == new_w {
        return image.clone();
    }

    let mut resized = Array3::<f32>::zeros((channels, new_h, new_w));
    let h_ratio = orig_h as f32 / new_h as f32;
    let w_ratio = orig_w as f32 / new_w as f32;

    for c in 0..channels {
        for y in 0..new_h {
            for x in 0..new_w {
                let src_y = ((y as f32 * h_ratio) as usize).min(orig_h - 1);
                let src_x = ((x as f32 * w_ratio) as usize).min(orig_w - 1);
                resized[[c, y, x]] = image[[c, src_y, src_x]];
            }
        }
    }
    resized
}

/// Normaliza pixels para [0, 1]
pub fn normalize_image(image: &Array3<f32>) -> Array3<f32> {
    image.mapv(|v| (v / 255.0).clamp(0.0, 1.0))
}

/// Normalização ImageNet: subtrai média e divide por desvio padrão por canal
pub fn normalize_imagenet(image: &Array3<f32>) -> Array3<f32> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let (channels, h, w) = image.dim();
    let mut out = Array3::<f32>::zeros((channels, h, w));
    for c in 0..channels.min(3) {
        for y in 0..h {
            for x in 0..w {
                out[[c, y, x]] = (image[[c, y, x]] - mean[c]) / std[c];
            }
        }
    }
    out
}

/// Flip horizontal
pub fn horizontal_flip(image: &Array3<f32>) -> Array3<f32> {
    let (c, h, w) = image.dim();
    let mut flipped = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                flipped[[ch, y, x]] = image[[ch, y, w - 1 - x]];
            }
        }
    }
    flipped
}

/// Ajuste de brilho multiplicativo
pub fn adjust_brightness(image: &Array3<f32>, factor: f32) -> Array3<f32> {
    image.mapv(|v| (v * factor).clamp(0.0, 1.0))
}

/// Adiciona ruído gaussiano
pub fn add_noise(image: &Array3<f32>, std_dev: f32) -> Array3<f32> {
    let mut rng = rand::thread_rng();
    image.mapv(|v| {
        let noise: f32 = rng.gen::<f32>() * 2.0 - 1.0; // [-1, 1]
        (v + noise * std_dev).clamp(0.0, 1.0)
    })
}

/// Aplica augmentation aleatório a uma imagem
pub fn augment_image(image: &Array3<f32>) -> Array3<f32> {
    let mut rng = rand::thread_rng();
    let mut img = image.clone();

    if rng.gen_bool(0.5) {
        img = horizontal_flip(&img);
    }

    let brightness_factor: f32 = rng.gen_range(0.85..1.15);
    img = adjust_brightness(&img, brightness_factor);

    if rng.gen_bool(0.3) {
        img = add_noise(&img, 0.02);
    }

    img
}

/// Expande o dataset com augmentation
pub fn augment_dataset(dataset: &FaceDataset, factor: usize) -> FaceDataset {
    println!("🔄 Augmentation: {}x (de {} para ~{} imagens)", factor, dataset.len(), dataset.len() * factor);
    let mut augmented = dataset.clone();

    for face_img in &dataset.images {
        for _ in 1..factor {
            let aug_data = augment_image(&face_img.data);
            augmented.images.push(FaceImage {
                data: aug_data,
                person_name: face_img.person_name.clone(),
                class_id: face_img.class_id,
                file_path: format!("{}_aug", face_img.file_path),
            });
        }
    }

    augmented
}

// ────────────────────────────────────────────────
//  Extração de features para KNN
// ────────────────────────────────────────────────

/// Extrai um vetor de features compacto de uma imagem [C, H, W]:
/// - Histograma por canal (16 bins)
/// - Pixels sub-amostrados a cada 8px
/// - Gradiente de magnitude médio
pub fn extract_features(image: &Array3<f32>) -> Vec<f32> {
    let (channels, height, width) = image.dim();
    let mut features = Vec::new();

    // 1) Histograma de 16 bins por canal
    for c in 0..channels {
        let mut hist = vec![0.0f32; 16];
        for y in 0..height {
            for x in 0..width {
                let v = image[[c, y, x]];
                let bin = ((v * 15.9999) as usize).min(15);
                hist[bin] += 1.0;
            }
        }
        let total = (height * width) as f32;
        for h in hist {
            features.push(h / total);
        }
    }

    // 2) Pixels sub-amostrados (passo 8)
    let step = 8;
    for c in 0..channels {
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                features.push(image[[c, y, x]]);
                x += step;
            }
            y += step;
        }
    }

    // 3) Média e desvio padrão globais
    let all_vals: Vec<f32> = image.iter().copied().collect();
    let mean = all_vals.iter().sum::<f32>() / all_vals.len() as f32;
    let variance = all_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / all_vals.len() as f32;
    features.push(mean);
    features.push(variance.sqrt());

    features
}

// ────────────────────────────────────────────────
//  Detecção de faces (placeholder / Viola-Jones simples)
// ────────────────────────────────────────────────

/// Detecta regiões de interesse em um frame bruto RGB [width*height*3].
/// Implementação simplificada: retorna a região central como candidato.
/// Em produção, substituir por Haar Cascade ou MTCNN.
pub fn detect_faces(raw_frame: &[u8], width: usize, height: usize) -> Vec<Array3<f32>> {
    if raw_frame.is_empty() || width == 0 || height == 0 {
        return vec![];
    }

    // Simulação: usa toda a imagem como ROI (região de interesse)
    // Em produção, implemente detecção real aqui
    let mut image = Array3::<f32>::zeros((3, height, width));
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            if idx + 2 < raw_frame.len() {
                image[[0, y, x]] = raw_frame[idx] as f32 / 255.0;
                image[[1, y, x]] = raw_frame[idx + 1] as f32 / 255.0;
                image[[2, y, x]] = raw_frame[idx + 2] as f32 / 255.0;
            }
        }
    }

    // Retorna a região central 80% da imagem como face candidata
    let margin_y = height / 10;
    let margin_x = width / 10;
    let roi_h = height - 2 * margin_y;
    let roi_w = width - 2 * margin_x;

    let roi = image
        .slice(s![.., margin_y..margin_y + roi_h, margin_x..margin_x + roi_w])
        .to_owned();

    let resized = resize_image(&roi, 128, 128);
    vec![resized]
}

/// Converte um frame RGB bruto em Array3<f32>
pub fn raw_frame_to_array(raw: &[u8], width: usize, height: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((3, height, width));
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            if idx + 2 < raw.len() {
                img[[0, y, x]] = raw[idx] as f32 / 255.0;
                img[[1, y, x]] = raw[idx + 1] as f32 / 255.0;
                img[[2, y, x]] = raw[idx + 2] as f32 / 255.0;
            }
        }
    }
    img
}

// ────────────────────────────────────────────────
//  Salvar imagem PPM
// ────────────────────────────────────────────────

/// Salva um frame RGB bruto como arquivo PPM P6
pub fn save_ppm(raw_rgb: &[u8], width: usize, height: usize, path: &Path) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(AppError::Io)?;
    }
    let mut file = fs::File::create(path).map_err(AppError::Io)?;
    writeln!(file, "P6\n{} {}\n255", width, height).map_err(AppError::Io)?;
    file.write_all(raw_rgb).map_err(AppError::Io)?;
    Ok(())
}

// ────────────────────────────────────────────────
//  Testes
// ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_add_image() {
        let mut ds = FaceDataset::new();
        let img = Array3::<f32>::zeros((3, 128, 128));
        ds.add_image(FaceImage {
            data: img,
            person_name: "Alice".into(),
            class_id: 0,
            file_path: "test.ppm".into(),
        });
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.num_classes(), 1);
        assert_eq!(ds.images[0].class_id, 0);
    }

    #[test]
    fn test_resize_image() {
        let img = Array3::<f32>::ones((3, 64, 64));
        let resized = resize_image(&img, 128, 128);
        assert_eq!(resized.dim(), (3, 128, 128));
    }

    #[test]
    fn test_normalize_image() {
        let img = Array3::<f32>::from_elem((3, 4, 4), 128.0);
        let norm = normalize_image(&img);
        for v in norm.iter() {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_horizontal_flip() {
        let mut img = Array3::<f32>::zeros((3, 4, 4));
        img[[0, 0, 0]] = 1.0;
        let flipped = horizontal_flip(&img);
        assert_eq!(flipped[[0, 0, 3]], 1.0);
        assert_eq!(flipped[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_extract_features_shape() {
        let img = Array3::<f32>::zeros((3, 128, 128));
        let feats = extract_features(&img);
        assert!(!feats.is_empty());
    }

    #[test]
    fn test_dataset_split() {
        let mut ds = FaceDataset::new();
        for i in 0..10 {
            ds.add_image(FaceImage {
                data: Array3::<f32>::zeros((3, 128, 128)),
                person_name: format!("person_{}", i % 2),
                class_id: 0,
                file_path: format!("img_{}.ppm", i),
            });
        }
        let (train, val) = ds.split(0.2);
        assert!(train.len() > 0);
        assert!(val.len() > 0);
        assert_eq!(train.len() + val.len(), 10);
    }
}