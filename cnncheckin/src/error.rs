// projeto cnncheckin
// src/error.rs — Tipos de erro unificados

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Erro de banco de dados: {0}")]
    Database(#[from] postgres::Error),

    #[error("Erro de pool de conexão: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("Erro de I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Erro de configuração: {0}")]
    Config(String),

    #[error("Erro de imagem: {0}")]
    Image(String),

    #[error("Modelo não treinado")]
    ModelNotTrained,

    #[error("Nenhuma face detectada")]
    NoFaceDetected,

    #[error("Dataset vazio ou insuficiente")]
    InsufficientData,

    #[error("Erro de câmera: {0}")]
    Camera(String),

    #[error("Erro genérico: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, AppError>;

impl From<Box<dyn std::error::Error>> for AppError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        AppError::Generic(e.to_string())
    }
}

impl From<String> for AppError {
    fn from(s: String) -> Self {
        AppError::Generic(s)
    }
}

impl From<&str> for AppError {
    fn from(s: &str) -> Self {
        AppError::Generic(s.to_string())
    }
}