use rand::Rng;
use rand_distr::{Distribution, Normal};
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use std::error::Error;

// Define a estrutura de um neurônio
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: rng.gen_range(-1.0..1.0),
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

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize, db_path: &str) -> Result<()> {
        let conn = Connection::open(db_path)?;
        conn.execute("DROP TABLE IF EXISTS training_data;", [])?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                data TEXT NOT NULL
            )",
            [],
        )?;

        for epoch in 0..epochs {
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

                // Backpropagation
                let mut errors = Vec::new();

                // Calcula o erro da camada de saída
                let output_activations = activations.last().unwrap();
                let output_errors: Vec<f64> = output_activations.iter().zip(targets).map(|(o, t)| (t - o) * o * (1.0 - o)).collect();
                errors.push(output_errors);

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
                            self.layers[l][j].weights[k] += learning_rate * errors[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * errors[l][j];
                    }
                }
            }

            let training_data_str = serde_json::to_string(&self).unwrap();
            conn.execute(
                "INSERT INTO training_data (epoch, data) VALUES (?1, ?2)",
                params![epoch as i32, training_data_str],
            )?;

            println!("Epoch {} completed.", epoch + 1);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let layer_sizes = &[2, 3, 1];
    let mut mlp = MLP::new(layer_sizes);

    let db_path = "consciencia/training_data.db";

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    mlp.train(&training_data, 0.3, 5000, db_path)?;

    println!("Rede treinada: {:?}", mlp);

    println!("Teste [0, 0]: {:?}", mlp.forward(&[0.0, 0.0]));
    println!("Teste [0, 1]: {:?}", mlp.forward(&[0.0, 1.0]));
    println!("Teste [1, 0]: {:?}", mlp.forward(&[1.0, 0.0]));
    println!("Teste [1, 1]: {:?}", mlp.forward(&[1.0, 1.0]));

    Ok(())
}