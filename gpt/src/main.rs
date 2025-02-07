use ndarray::{Array2, Axis, array};
use rand::Rng;

// Activation functions and their derivatives
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

// MLP struct definition
struct MLP {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    activation_functions: Vec<fn(f64) -> f64>,
    activation_derivatives: Vec<fn(f64) -> f64>,
}

impl MLP {
    // Constructor to initialize the MLP
    fn new(layer_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activation_functions = Vec::new();
        let mut activation_derivatives = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];

            // Weight initialization using He initialization for ReLU
            let bound = (2.0 / cols as f64).sqrt();
            weights.push(Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-bound..bound)));
            biases.push(Array2::zeros((rows, 1)));

            // Assign activation functions and derivatives
            if i < layer_sizes.len() - 2 {
                activation_functions.push(relu as fn(f64) -> f64);
                activation_derivatives.push(relu_derivative as fn(f64) -> f64);
            } else {
                activation_functions.push(sigmoid as fn(f64) -> f64);
                activation_derivatives.push(sigmoid_derivative as fn(f64) -> f64);
            }
        }

        MLP {
            weights,
            biases,
            activation_functions,
            activation_derivatives,
        }
    }

    // Forward pass through the network
    fn forward(&self, input: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut activations = vec![input.clone()];
        let mut z_values = Vec::new();

        for i in 0..self.weights.len() {
            let z = activations[i].dot(&self.weights[i].t()) + &self.biases[i];
            z_values.push(z.clone());
            let a = z.mapv(self.activation_functions[i]);
            activations.push(a);
        }

        activations
    }

    // Training the network using backpropagation
    fn train(&mut self, training_data: &[(Array2<f64>, Array2<f64>)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for (input, target) in training_data.iter() {
                // Forward pass
                let activations = self.forward(input);
                let output = activations.last().unwrap();

                // Calculate error (MSE)
                let error = output - target;
                total_error += error.mapv(|x| x.powi(2)).sum();

                // Backward pass
                let mut deltas = Vec::new();
                let output_delta = error * output.mapv(self.activation_derivatives.last().unwrap());
                deltas.push(output_delta);

                // Propagate error backward
                for l in (0..self.weights.len() - 1).rev() {
                    let delta = deltas.last().unwrap().dot(&self.weights[l + 1])
                        * activations[l + 1].mapv(self.activation_derivatives[l]);
                    deltas.push(delta);
                }

                deltas.reverse();

                // Update weights and biases
                for l in 0..self.weights.len() {
                    let weight_gradient = deltas[l].t().dot(&activations[l]);
                    self.weights[l] = &self.weights[l] - &(learning_rate * weight_gradient);

                    let bias_gradient = deltas[l].sum_axis(Axis(1)).insert_axis(Axis(1));
                    self.biases[l] = &self.biases[l] - &(learning_rate * bias_gradient);
                }
            }

            // Print average error for the epoch
            let avg_error = total_error / training_data.len() as f64;
            if epoch % 1000 == 0 {
                println!("Epoch {}: Average Error = {:.6}", epoch, avg_error);
            }

            // Early stopping
            if avg_error < 0.001 {
                println!("Converged at epoch {}", epoch);
                break;
            }
        }
    }
}

fn main() {
    // XOR training data
    let training_data = vec![
        (array![[0.0, 0.0]], array![[0.0]]),
        (array![[0.0, 1.0]], array![[1.0]]),
        (array![[1.0, 0.0]], array![[1.0]]),
        (array![[1.0, 1.0]], array![[0.0]]),
    ];

    // Define the MLP architecture
    let layer_sizes = &[2, 4, 1];
    let mut mlp = MLP::new(layer_sizes);

    // Train the MLP
    mlp.train(&training_data[..], 0.1, 10000);

    // Test the trained MLP
    println!("\nTest Results:");
    for (input, _) in &training_data {
        let output = mlp.forward(input);
        let predicted = output.last().unwrap();
        println!("Input: {:?}, Predicted Output: {:.4}", input, predicted);
    }
}