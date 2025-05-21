// src/mlp/previsao_lstm.rs
use burn::{
    module::Module,
    nn::{
        loss::MSELoss,
        rnn::{LSTM, LSTMConfig},
        Linear, LinearConfig,
    },
    tensor::{
        backend::Backend,
        Tensor,
        Data,
        Shape,
    },
    train::{
        LearnerBuilder,
        metric::LossMetric,
    },
};
use ndarray::{Array2, ArrayView2};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LSTMError {
    #[error("Invalid data format: {0}")]
    InvalidData(String),
}

pub struct LSTMModel<B: Backend> {
    lstm: LSTM<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config_lstm = LSTMConfig::new(5, 64); // 5 features
        let lstm = LSTM::new(device, config_lstm);

        let config_linear = LinearConfig::new(64, 1);
        let linear = Linear::new(device, config_linear);

        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let outputs = self.lstm.forward(inputs);
        let last_output = outputs.select(1, -1);
        self.linear.forward(last_output)
    }
}