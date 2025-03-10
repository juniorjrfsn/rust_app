use ndarray::{Array1, Array2, Array, arr1, prelude::*};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use std::collections::HashMap;
use itertools::Itertools;

// Define the MLP struct
pub struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    bias_hidden: Array1<f64>,
    bias_output: Array1<f64>,
    learning_rate: f64,
}

impl MLP {
    // Constructor with He initialization
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let weights_input_hidden = Array::random(
            (input_size, hidden_size),
            Normal::new(0.0, (2.0 / input_size as f64).sqrt()).unwrap()
        );
        
        let weights_hidden_output = Array::random(
            (hidden_size, output_size),
            Normal::new(0.0, (2.0 / hidden_size as f64).sqrt()).unwrap()
        );

        MLP {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden: Array1::zeros(hidden_size),
            bias_output: Array1::zeros(output_size),
            learning_rate,
        }
    }

    // Forward pass
    pub fn forward(&self, inputs: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden_inputs = self.weights_input_hidden.t().dot(inputs) + &self.bias_hidden;
        let hidden_outputs = hidden_inputs.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        
        let output_inputs = self.weights_hidden_output.t().dot(&hidden_outputs) + &self.bias_output;
        let output = softmax(&output_inputs);
        
        (hidden_outputs, output)
    }

    // Training step
    pub fn train(&mut self, inputs: &Array1<f64>, target: &Array1<f64>) {
        let (hidden_outputs, output) = self.forward(inputs);
        let output_errors = &output - target;
        
        let sigmoid_derivative = hidden_outputs.mapv(|x| x * (1.0 - x));
        let hidden_errors = self.weights_hidden_output.dot(output_errors) * sigmoid_derivative;

        // Update output layer weights and biases
        self.weights_hidden_output -= 
            &(hidden_outputs.outer(output_errors) * self.learning_rate);
        self.bias_output -= &(output_errors * self.learning_rate);

        // Update input layer weights and biases
        self.weights_input_hidden -= 
            &(inputs.outer(&hidden_errors) * self.learning_rate);
        self.bias_hidden -= &(hidden_errors * self.learning_rate);
    }
}

// Softmax activation function
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let exp_x = x.mapv(|v| v.exp());
    exp_x / exp_x.sum()
}

fn main() {
    // Training data
    let training_data: Vec<([u8; 3], &str)> = vec![
        ([255, 0, 127], "Rose"),
        ([127, 0, 0], "Vermelho"),
        ([255, 0, 0], "Vermelho"),
        ([255, 127, 0], "Laranja"),
        ([127, 127, 0], "Amarelo"),
        ([255, 255, 0], "Amarelo"),
        ([127, 255, 0], "Primavera"),
        ([0, 127, 0], "Verde"),
        ([0, 255, 0], "Verde"),
        ([0, 255, 127], "Turquesa"),
        ([0, 127, 127], "Ciano"),
        ([0, 255, 255], "Ciano"),
        ([0, 127, 255], "Cobalto"),
        ([0, 0, 127], "Azul"),
        ([0, 0, 255], "Azul"),
        ([127, 0, 255], "Violeta"),
        ([127, 0, 127], "Magenta"),
        ([255, 0, 255], "Magenta"),
        ([0, 0, 0], "Preto"),
        ([127, 127, 127], "Cinza"),
        ([255, 255, 255], "Branco"),
    ];

    // Prepare color mappings
    let colors: Vec<&str> = training_data.iter()
        .map(|(_, c)| *c)
        .unique()
        .sorted()
        .collect();
    let color_idx: HashMap<&str, usize> = colors.iter().enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // Normalize data and create one-hot encodings
    let normalized_data: Vec<(Array1<f64>, Array1<f64>)> = training_data.iter()
        .map(|&(rgb, color)| {
            let inputs = arr1(&[
                rgb[0] as f64 / 255.0,
                rgb[1] as f64 / 255.0,
                rgb[2] as f64 / 255.0,
            ]);
            
            let mut target = Array1::zeros(colors.len());
            target[color_idx[color]] = 1.0;
            
            (inputs, target)
        })
        .collect();

    // Initialize network
    let mut mlp = MLP::new(3, 32, colors.len(), 0.01);
    let epochs = 10000;

    // Training loop
    for epoch in 0..epochs {
        for (inputs, target) in &normalized_data {
            mlp.train(inputs, target);
        }

        // Print loss every 1000 epochs
        if epoch % 1000 == 0 {
            let loss: f64 = normalized_data.iter()
                .map(|(i, t)| {
                    let (_, o) = mlp.forward(i);
                    o.iter()
                        .zip(t.iter())
                        .map(|(o, t)| (o - t).powi(2))
                        .sum::<f64>()
                })
                .sum();
            println!("Epoch {:5} | Loss: {:.4}", epoch, loss);
        }
    }

    // Test data
    let test_data = vec![
        [200, 0, 70],
        [165, 156, 159],
    ];

    println!("\n================ TESTE ===============");
    for rgb in test_data {
        let inputs = arr1(&[
            rgb[0] as f64 / 255.0,
            rgb[1] as f64 / 255.0,
            rgb[2] as f64 / 255.0,
        ]);

        let (_, output) = mlp.forward(&inputs);
        let max_idx = output.argmax().unwrap();
        let predicted = colors[max_idx];
        let confidence = output[max_idx] * 100.0;

        println!(
            "Cor RGB: {:?} | Previsão: {:<10} | Confiança: {:.1}%",
            rgb, predicted, confidence
        );
    }
}