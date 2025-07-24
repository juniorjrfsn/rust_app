use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Lstm, LstmConfig},
    tensor::{backend::Backend, Tensor},
};
use crate::mlp::common::AppError;

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Lstm<B>,
    linear: Linear<B>,
}

impl<B: Backend> LSTMModel<B> {
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let config_lstm = LstmConfig::new(input_size, hidden_size, true);
        let lstm = config_lstm.init(device);
        let config_linear = LinearConfig::new(hidden_size * 2, 1);
        let linear = config_linear.init(device);
        Self { lstm, linear }
    }

    pub fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 2> {
        let (outputs, _) = self.lstm.forward(inputs, None);
        let last_output = outputs.slice([0..outputs.dims()[0], outputs.dims()[1] - 1..outputs.dims()[1]]);
        let last_output_2d = last_output.reshape([last_output.dims()[0], last_output.dims()[2]]);
        self.linear.forward(last_output_2d)
    }
}