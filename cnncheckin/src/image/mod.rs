// file: src/image/mod.rs
// Módulo de processamento de imagens refatorado

mod dataset;
mod preprocessing;
mod augmentation;
mod detection;

pub use dataset::{FaceDataset, FaceImage, DataLoader};
pub use preprocessing::{ImagePreprocessor, NormalizationStrategy};
pub use augmentation::ImageAugmenter;
pub use detection::FaceDetector;

use ndarray::Array3;
use crate::error::Result;

/// Representa uma imagem processada
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Dados da imagem [channels, height, width]
    pub data: Array3<f32>,
    /// Largura original
    pub original_width: usize,
    /// Altura original
    pub original_height: usize,
}

impl ProcessedImage {
    pub fn new(data: Array3<f32>, original_width: usize, original_height: usize) -> Self {
        Self {
            data,
            original_width,
            original_height,
        }
    }
    
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.data.dim()
    }
    
    pub fn channels(&self) -> usize {
        self.data.dim().0
    }
    
    pub fn height(&self) -> usize {
        self.data.dim().1
    }
    
    pub fn width(&self) -> usize {
        self.data.dim().2
    }
}

/// Pipeline de processamento de imagens
pub struct ImagePipeline {
    preprocessor: ImagePreprocessor,
    detector: Option<FaceDetector>,
    augmenter: Option<ImageAugmenter>,
}

impl ImagePipeline {
    pub fn new(target_size: (usize, usize)) -> Self {
        Self {
            preprocessor: ImagePreprocessor::new(target_size),
            detector: None,
            augmenter: None,
        }
    }
    
    pub fn with_detection(mut self) -> Self {
        self.detector = Some(FaceDetector::new());
        self
    }
    
    pub fn with_augmentation(mut self) -> Self {
        self.augmenter = Some(ImageAugmenter::new());
        self
    }
    
    /// Processa uma imagem bruta
    pub fn process(&self, raw_data: &[u8], width: usize, height: usize) -> Result<Vec<ProcessedImage>> {
        // Converter para Array3
        let image = self.raw_to_array(raw_data, width, height)?;
        
        // Detectar faces se habilitado
        let faces = if let Some(detector) = &self.detector {
            detector.detect(&image)?
        } else {
            vec![image]
        };
        
        // Preprocessar cada face
        let mut processed = Vec::new();
        for face in faces {
            let proc = self.preprocessor.process(&face)?;
            
            // Augmentar se habilitado
            if let Some(augmenter) = &self.augmenter {
                let augmented = augmenter.augment(&proc.data)?;
                for aug in augmented {
                    processed.push(ProcessedImage::new(aug, width, height));
                }
            } else {
                processed.push(proc);
            }
        }
        
        Ok(processed)
    }
    
    fn raw_to_array(&self, raw_data: &[u8], width: usize, height: usize) -> Result<Array3<f32>> {
        use crate::error::AppError;
        
        if raw_data.len() < width * height * 3 {
            return Err(AppError::image("Dados de imagem insuficientes"));
        }
        
        let mut image = Array3::<f32>::zeros((3, height, width));
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                
                if idx + 2 < raw_data.len() {
                    image[[0, y, x]] = raw_data[idx] as f32 / 255.0;
                    image[[1, y, x]] = raw_data[idx + 1] as f32 / 255.0;
                    image[[2, y, x]] = raw_data[idx + 2] as f32 / 255.0;
                }
            }
        }
        
        Ok(image)
    }
}

/// Utilitários para manipulação de imagens
pub mod utils {
    use super::*;
    use std::fs;
    use std::path::Path;
    
    /// Salva imagem em formato PPM
    pub fn save_ppm(image: &Array3<f32>, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        use crate::error::AppError;
        
        let (channels, height, width) = image.dim();
        
        if channels != 3 {
            return Err(AppError::image("Imagem deve ter 3 canais"));
        }
        
        let mut file = fs::File::create(path)?;
        
        // Cabeçalho PPM
        writeln!(file, "P6")?;
        writeln!(file, "{} {}", width, height)?;
        writeln!(file, "255")?;
        
        // Dados de pixel
        for y in 0..height {
            for x in 0..width {
                let r = (image[[0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (image[[1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (image[[2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                
                file.write_all(&[r, g, b])?;
            }
        }
        
        Ok(())
    }
    
    /// Carrega imagem PPM
    pub fn load_ppm(path: impl AsRef<Path>) -> Result<Array3<f32>> {
        use crate::error::AppError;
        
        let content = fs::read(path)?;
        let mut lines = content.split(|&b| b == b'\n');
        
        // Magic number
        let magic = lines.next().ok_or_else(|| AppError::image("PPM inválido"))?;
        if magic != b"P6" {
            return Err(AppError::image("Apenas formato PPM P6 é suportado"));
        }
        
        // Pular comentários
        let mut dims_line = lines.next().ok_or_else(|| AppError::image("PPM inválido"))?;
        while dims_line.starts_with(b"#") {
            dims_line = lines.next().ok_or_else(|| AppError::image("PPM inválido"))?;
        }
        
        // Dimensões
        let dims = String::from_utf8(dims_line.to_vec())
            .map_err(|_| AppError::image("Dimensões inválidas"))?;
        let parts: Vec<&str> = dims.trim().split_whitespace().collect();
        
        if parts.len() != 2 {
            return Err(AppError::image("Formato de dimensões incorreto"));
        }
        
        let width: usize = parts[0].parse()
            .map_err(|_| AppError::image("Largura inválida"))?;
        let height: usize = parts[1].parse()
            .map_err(|_| AppError::image("Altura inválida"))?;
        
        // Valor máximo
        let max_line = lines.next().ok_or_else(|| AppError::image("PPM inválido"))?;
        let max_val: u8 = String::from_utf8(max_line.to_vec())
            .map_err(|_| AppError::image("Valor máximo inválido"))?
            .trim()
            .parse()
            .map_err(|_| AppError::image("Valor máximo inválido"))?;
        
        // Dados de pixel
        let header_size = content.len() - lines.as_slice().len();
        let pixel_data = &content[header_size..];
        
        if pixel_data.len() < width * height * 3 {
            return Err(AppError::image("Dados de pixel insuficientes"));
        }
        
        let mut image = Array3::<f32>::zeros((3, height, width));
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                
                image[[0, y, x]] = pixel_data[idx] as f32 / max_val as f32;
                image[[1, y, x]] = pixel_data[idx + 1] as f32 / max_val as f32;
                image[[2, y, x]] = pixel_data[idx + 2] as f32 / max_val as f32;
            }
        }
        
        Ok(image)
    }
    
    /// Redimensiona imagem usando nearest neighbor
    pub fn resize(image: &Array3<f32>, target_width: usize, target_height: usize) -> Result<Array3<f32>> {
        let (channels, orig_height, orig_width) = image.dim();
        
        if orig_width == target_width && orig_height == target_height {
            return Ok(image.clone());
        }
        
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processed_image() {
        let data = Array3::<f32>::zeros((3, 128, 128));
        let img = ProcessedImage::new(data, 640, 480);
        
        assert_eq!(img.channels(), 3);
        assert_eq!(img.height(), 128);
        assert_eq!(img.width(), 128);
        assert_eq!(img.original_width, 640);
        assert_eq!(img.original_height, 480);
    }
    
    #[test]
    fn test_image_pipeline() {
        let pipeline = ImagePipeline::new((128, 128));
        
        let raw_data = vec![128u8; 640 * 480 * 3];
        let result = pipeline.process(&raw_data, 640, 480);
        
        assert!(result.is_ok());
        let processed = result.unwrap();
        assert!(!processed.is_empty());
    }
    
    #[test]
    fn test_resize() {
        let image = Array3::<f32>::ones((3, 64, 64));
        let resized = utils::resize(&image, 128, 128).unwrap();
        
        assert_eq!(resized.dim(), (3, 128, 128));
    }
}