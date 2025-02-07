use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let bound = (6.0 / (num_inputs + 1) as f64).sqrt();
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-bound..bound)).collect(),
            bias: rng.gen_range(-bound..bound),
        }
    }

    fn activate(&self, inputs: &[f64], activation_fn: &str) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        match activation_fn {
            "tanh" => sum.tanh(),
            "sigmoid" => 1.0 / (1.0 + (-sum).exp()),
            _ => panic!("Unsupported activation function"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MLP {
    layers: Vec<Vec<Neuron>>,
}

impl MLP {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let num_neurons = layer_sizes[i + 1];
            let num_inputs = layer_sizes[i];
            let layer = (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect();
            layers.push(layer);
        }
        MLP { layers }
    }

    fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![inputs.to_vec()];
        for (i, layer) in self.layers.iter().enumerate() {
            let mut new_activations = Vec::new();
            let activation_fn = if i == self.layers.len() - 1 {
                "sigmoid"
            } else {
                "tanh"
            };
            for neuron in layer {
                new_activations.push(neuron.activate(&activations.last().unwrap(), activation_fn));
            }
            activations.push(new_activations);
        }
        activations
    }

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (inputs, targets) in training_data.iter() {
                // Forward pass
                let activations = self.forward(inputs);
                let output = activations.last().unwrap();

                // Calculate output error (using MSE for simplicity)
                let output_errors: Vec<f64> = output
                    .iter()
                    .zip(targets)
                    .map(|(o, t)| t - o) // Just the difference for MSE
                    .collect();
                total_error += output_errors.iter().map(|e| e * e).sum::<f64>(); // Sum of squared errors

                // Backpropagation
                let mut deltas = vec![output_errors
                    .iter()
                    .zip(output.iter())
                    .map(|(err, out)| err * out * (1.0 - out)) // Sigmoid derivative
                    .collect::<Vec<f64>>()]; // Output layer deltas

                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_deltas = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut delta_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            delta_sum += deltas.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        let activation = activations[l + 1][j];
                        layer_deltas.push(delta_sum * (1.0 - activation * activation)); // Tanh derivative
                    }
                    deltas.push(layer_deltas);
                }
                deltas.reverse();

                // Update weights and biases
                for l in 0..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        for k in 0..self.layers[l][j].weights.len() {
                            self.layers[l][j].weights[k] +=
                                learning_rate * deltas[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * deltas[l][j];
                    }
                }
            }
            let avg_error = total_error / training_data.len() as f64;
            println!("Epoch {}: Average Error = {:.6}", epoch + 1, avg_error);
        }
    }
}

fn main() {
    let layer_sizes = &[2, 3, 1];
    let mut mlp = MLP::new(layer_sizes);
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    mlp.train(&training_data, 0.1, 10000);

    println!("\nTest Results:");
    println!(
        "Input: [0, 0], Predicted Output: {:?}",
        mlp.forward(&[0.0, 0.0]).last().unwrap()
    );
    println!(
        "Input: [0, 1], Predicted Output: {:?}",
        mlp.forward(&[0.0, 1.0]).last().unwrap()
    );
    println!(
        "Input: [1, 0], Predicted Output: {:?}",
        mlp.forward(&[1.0, 0.0]).last().unwrap()
    );
    println!(
        "Input: [1, 1], Predicted Output: {:?}",
        mlp.forward(&[1.0, 1.0]).last().unwrap()
    );
}