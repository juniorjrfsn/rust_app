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
        let bound = (6.0 / (num_inputs + 1) as f64).sqrt(); // Xavier initialization
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-bound..bound)).collect(),
            bias: rng.gen_range(-bound..bound),
        }
    }

    fn activate(&self, inputs: &[f64], activation_fn: &str) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        match activation_fn {
            "linear" => sum, // Linear activation for output layer
            "tanh" => sum.tanh(),
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
                "linear" // Linear activation for output layer
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

    fn train(&mut self, training_data: &[(Vec<f64>, f64)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (inputs, target) in training_data.iter() {
                // Forward pass
                let activations = self.forward(inputs);
                let output = activations.last().unwrap()[0];
                // Calculate error (MSE)
                let error = target - output;
                total_error += error * error;
                // Backpropagation
                let mut deltas = vec![vec![error]]; // Delta for output layer
                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_deltas = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut delta_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            delta_sum += deltas.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        let activation = activations[l + 1][j];
                        layer_deltas.push(delta_sum * (1.0 - activation * activation)); // Derivative of tanh
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
            if epoch % 1000 == 0 || epoch == epochs - 1 {
                println!("Epoch {}: Average Error = {:.6}", epoch + 1, avg_error);
            }
        }
    }
}

fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let normalized: Vec<f64> = data.iter().map(|x| (x - min) / (max - min)).collect();
    (normalized, min, max)
}

fn main() {
    // Dados históricos: dias (x) e bid prices (y)
    let days: Vec<f64> = (0..30).map(|x| x as f64).collect(); // Dias 0 a 29
    let bid_prices: Vec<f64> = vec![
        1.0411, 1.0387, 1.0406, 1.0381, 1.0351, 1.0271, 1.0435, 1.0434, 1.0399, 1.0428,
        1.0431, 1.0495, 1.0487, 1.0523, 1.0522, 1.0421, 1.0415, 1.0434, 1.0426, 1.0285,
        1.0331, 1.0330, 1.0305, 1.0355, 1.0309, 1.0278, 1.0247, 1.0313, 1.0216, 1.0329,
    ];

    // Normalizar os dados
    let (normalized_prices, min_price, max_price) = normalize(&bid_prices);

    // Criar dados de treinamento
    let window_size = 5; // Usar os últimos 5 dias como entrada
    let mut training_data = Vec::new();
    for i in window_size..normalized_prices.len() {
        let inputs: Vec<f64> = normalized_prices[i - window_size..i].to_vec();
        let target = normalized_prices[i];
        training_data.push((inputs, target));
    }

    // Criar e treinar a MLP
    let layer_sizes = &[window_size, 10, 1]; // Camada de entrada (5), oculta (10), saída (1)
    let mut mlp = MLP::new(layer_sizes);
    mlp.train(&training_data, 0.01, 10000);

    // Prever o próximo valor (dia 30)
    let last_inputs: Vec<f64> = normalized_prices[normalized_prices.len() - window_size..].to_vec();
    let normalized_prediction = mlp.forward(&last_inputs).last().unwrap()[0];

    // Desnormalizar a previsão
    let denormalized_prediction = normalized_prediction * (max_price - min_price) + min_price;

    println!(
        "Previsão para o bid price em 08/02/2025: {:.4}",
        denormalized_prediction
    );
}