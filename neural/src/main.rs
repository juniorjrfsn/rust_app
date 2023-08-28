use ndarray::{Array, Array1, Array2, Axis, aview1};
use rand::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use ndarray::{Array, Array1, Array2, Axis};
struct NeuralNetwork {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((hidden_size, input_size), |_| rng.gen::<f64>());
        let biases = Array::zeros(hidden_size);
        NeuralNetwork { weights, biases }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let hidden_input = self.weights.dot(input) + &self.biases;
        hidden_input.mapv(|x| self.sigmoid(x))
    }
}

fn main() {
    let input_size = 2;
    let hidden_size = 2;
    let output_size = 1;

    let nn = NeuralNetwork::new(input_size, hidden_size, output_size);

    let training_data = vec![
        (Array::from_vec(vec![0.0, 0.0]), Array::from_vec(vec![0.0, 0.6])),
        (Array::from_vec(vec![7.0, 2.0]), Array::from_vec(vec![1.0, 0.0])),
        (Array::from_vec(vec![4.0, 2.0]), Array::from_vec(vec![1.0, 0.0])),
        (Array::from_vec(vec![1.0, 3.0]), Array::from_vec(vec![2.0, 0.0])),
        (Array::from_vec(vec![5.0, 4.5]), Array::from_vec(vec![2.0, 0.0])),
    ];

    // Training loop
    for epoch in 0..10000 {
        let mut total_loss = 0.0;

        for (input, target) in &training_data {
            let hidden_output = nn.forward(input);
            let loss = (hidden_output[0] - target[0]).powi(2);
            total_loss += loss;
            //println!("Loss: {:?}",   loss);
            // Backpropagation (not implemented in this example)
            // Adjust weights and biases using gradient descent
        }

        if epoch % 1000 == 0 {
            println!("Epoch: {}, Loss: {}", epoch, total_loss);
        }
    }

    let input = Array::from_vec(vec![0.0, 0.0]);
    let prediction = nn.forward(&input);
    println!("=========================================");
    println!("Input: {:?}", input);
    println!("Prediction: {:?} ::", prediction);
}