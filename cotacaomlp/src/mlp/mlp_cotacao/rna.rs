use rand::prelude::*;
use serde::{Serialize, Deserialize};
use bincode;
use std::error::Error;

// Ativações e derivadas
fn apply_activation(x: f64, activation: &str) -> f64 {
    match activation {
        "relu" => x.max(0.0),
        "tanh" => x.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        _ => x,
    }
}

fn activation_derivative(x: f64, activation: &str) -> f64 {
    match activation {
        "relu" => if x > 0.0 { 1.0 } else { 0.0 },
        "tanh" => 1.0 - x.tanh().powi(2),
        "sigmoid" => {
            let s = apply_activation(x, "sigmoid");
            s * (1.0 - s)
        }
        _ => 1.0,
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let mut rng = thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let biases = vec![0.0; output_size];
        DenseLayer { weights, biases }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(w, b)| {
                w.iter().zip(inputs).map(|(x, i)| x * i).sum::<f64>() + b
            })
            .collect()
    }

    pub fn update_weights(&mut self, grads: &[Vec<f64>], lr: f64) {
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, weight) in row.iter_mut().enumerate() {
                *weight -= lr * grads[i][j];
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MLP {
    pub layers: Vec<DenseLayer>,
}

impl MLP {
    pub fn new(sizes: &[usize]) -> Self {
        let layers = sizes
            .windows(2)
            .map(|s| DenseLayer::new(s[0], s[1]))
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, input: &[f64], activation: &str) -> Vec<f64> {
        let mut out = input.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
            out = out.iter().map(|&x| apply_activation(x, activation)).collect();
        }
        out
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        labels: &[f64],
        epochs: usize,
        learning_rate: f64,
        activation: &str,
    ) -> Result<(), Box<dyn Error>> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, label) in inputs.iter().zip(labels.iter()) {
                // Forward
                let mut activations = vec![input.clone()];
                for layer in &self.layers {
                    let out = layer.forward(activations.last().unwrap());
                    activations.push(out.iter().map(|x| apply_activation(*x, activation)).collect());
                }

                let prediction = activations.last().unwrap()[0];
                let loss = (prediction - label).powi(2);
                total_loss += loss;

                // Backward
                let mut delta = 2.0 * (prediction - label);
                for i in (0..self.layers.len()).rev() {
                    let act = &activations[i + 1];
                    let inp = &activations[i];
                    let grad = act
                        .iter()
                        .enumerate()
                        .map(|(j, &a)| {
                            delta * activation_derivative(a, activation) * inp[j]
                        })
                        .collect::<Vec<_>>();
                    let grads = self.layers[i].weights.iter().map(|_| grad.clone()).collect();
                    self.layers[i].update_weights(&grads, learning_rate);

                    delta = self.layers[i]
                        .weights
                        .iter()
                        .flatten()
                        .zip(act.iter())
                        .map(|(w, a)| w * activation_derivative(*a, activation))
                        .sum();
                }
            }
            println!("Epoch {}: Loss {:.4}", epoch + 1, total_loss / inputs.len() as f64);
        }
        Ok(())
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, Box<dyn Error>> {
        Ok(bincode::deserialize(data)?)
    }
}