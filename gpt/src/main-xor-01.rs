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

    fn activate(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
        // Usando sigmoide como função de ativação
        1.0 / (1.0 + (-sum - self.bias).exp())
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

    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut activations = inputs.to_vec();
        for layer in &self.layers {
            let mut new_activations = Vec::new();
            for neuron in layer {
                new_activations.push(neuron.activate(&activations));
            }
            activations = new_activations;
        }
        activations
    }

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for (inputs, targets) in training_data.iter() {
                // Forward pass
                let mut activations = vec![inputs.clone()];
                for layer in &self.layers {
                    let mut new_activations = Vec::new();
                    for neuron in layer {
                        new_activations.push(neuron.activate(&activations.last().unwrap()));
                    }
                    activations.push(new_activations);
                }

                // Calcula o erro da camada de saída
                let output_activations = activations.last().unwrap();
                let output_errors: Vec<f64> = output_activations
                    .iter()
                    .zip(targets)
                    .map(|(o, t)| (t - o) * o * (1.0 - o))
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
                        layer_errors.push(error_sum * activation * (1.0 - activation));
                    }
                    errors.push(layer_errors);
                }
                errors.reverse();

                // Atualiza os pesos e biases
                for l in 0..self.layers.len() {
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

            // Critério de parada mais estrito
            if avg_error < 0.0001 {
                println!("Converged at epoch {}", epoch + 1);
                break;
            }
        }
    }
}

fn main() {
    // Exemplo de uso
    let layer_sizes = &[2, 4, 1]; // Rede com 2 entradas, 4 neurônios na camada oculta e 1 saída
    let mut mlp = MLP::new(layer_sizes);

    // Dados de treinamento (exemplo simples com XOR lógico)
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Treinamento
    mlp.train(&training_data, 0.1, 10000); // Taxa de aprendizado reduzida

    // Teste
    println!("\nTest Results:");
    println!("Input: [0, 0], Predicted Output: {:?}", mlp.forward(&[0.0, 0.0]));
    println!("Input: [0, 1], Predicted Output: {:?}", mlp.forward(&[0.0, 1.0]));
    println!("Input: [1, 0], Predicted Output: {:?}", mlp.forward(&[1.0, 0.0]));
    println!("Input: [1, 1], Predicted Output: {:?}", mlp.forward(&[1.0, 1.0]));
}