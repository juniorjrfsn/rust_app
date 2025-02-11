use rand::Rng;
use serde::{Deserialize, Serialize};

// Define a estrutura de um neurônio
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Inicialização de Xavier
        let bound = (6.0 / (num_inputs + 1) as f64).sqrt();
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-bound..bound)).collect(),
            bias: rng.gen_range(-bound..bound),
        }
    }

    fn activate(&self, inputs: &[f64], activation_fn: &str) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        match activation_fn {
            "tanh" => sum.tanh(), // Tanh
            "sigmoid" => 1.0 / (1.0 + (-sum).exp()), // Sigmoid
            _ => panic!("Unsupported activation function"),
        }
    }
}

// Define a estrutura de uma rede neural multicamadas (MLP)
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
                "sigmoid" // Sigmoid na camada de saída
            } else {
                "tanh" // Tanh nas camadas ocultas
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

                // Calcula o erro da camada de saída
                let output_errors: Vec<f64> = activations
                    .last().unwrap()
                    .iter()
                    .zip(targets)
                    .map(|(o, t)| (t - o) * o * (1.0 - o)) // Derivada da sigmoide
                    .collect();
                total_error += output_errors.iter().map(|e| e.powi(2)).sum::<f64>();

                // Backpropagation
                let mut errors = vec![output_errors];

                // Propaga os erros para as camadas anteriores
                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_errors = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut error_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            error_sum += errors.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        let activation = activations[l + 1][j];
                        layer_errors.push(error_sum * (1.0 - activation.powi(2))); // Derivada da Tanh
                    }
                    errors.push(layer_errors);
                }
                errors.reverse();

                // Atualiza os pesos e biases
                for l in 00..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        for k in 0..self.layers[l][j].weights.len() {
                            self.layers[l][j].weights[k] +=
                                learning_rate * errors[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * errors[l][j];
                    }
                }
            }

            let avg_error = total_error / training_data.len() as f64;
            println!("Epoch {}: Average Error = {:.6}", epoch + 1, avg_error);

            // Early stopping
            if avg_error < 0.001 {
                println!("Converged at epoch {}", epoch + 1);
                break;
            }
        }
    }
}

fn main() {
    // Exemplo de uso
    let layer_sizes = &[2, 3, 1]; // Rede com 2 entradas, 3 neurônios na camada oculta e 1 saída
    let mut mlp = MLP::new(layer_sizes);

    // Dados de treinamento (exemplo simples com XOR lógico)
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Treinamento
    mlp.train(&training_data, 0.5, 10000); // Taxa de aprendizado maior

    // Teste
    println!("\nTest Results:");
    println!("Input: [0, 0], Predicted Output: {:?}", mlp.forward(&[0.0, 0.0]).last().unwrap());
    println!("Input: [0, 1], Predicted Output: {:?}", mlp.forward(&[0.0, 1.0]).last().unwrap());
    println!("Input: [1, 0], Predicted Output: {:?}", mlp.forward(&[1.0, 0.0]).last().unwrap());
    println!("Input: [1, 1], Predicted Output: {:?}", mlp.forward(&[1.0, 1.0]).last().unwrap());
}

// cd gpt
// cargo run
// cargo run --bin gpt
// cargo.exe "run", "--package", "gpt", "--bin", "gpt"