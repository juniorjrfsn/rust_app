// projeto: lstmrnntrain
// file: src/neural/mod.rs
// Module declarations for the neural network training system

pub mod utils;    // Utility functions, optimizers, and error handling
pub mod storage;  // Model and metrics storage logic for PostgreSQL
pub mod metrics;  // Training metrics definitions and calculations
pub mod model;    // Multi-model neural network implementations (LSTM, RNN, MLP, CNN)
pub mod data;     // Data loading and preprocessing functionality

// Re-export commonly used items for convenience
pub use model::{ModelType, NeuralNetwork};
pub use metrics::TrainingMetrics;
pub use utils::{TrainingError, AdamOptimizer};
pub use data::{DataLoader, FeatureStats, StockRecord};
pub use storage::{save_model_to_postgres, load_model_from_postgres};