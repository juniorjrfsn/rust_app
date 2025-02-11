use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Estrutura para representar um neurônio
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    // Função para criar um novo neurônio
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let bound = (6.0 / (num_inputs + 1) as f64).sqrt(); // Limite para inicialização de Xavier
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-bound..bound)).collect(),
            bias: rng.gen_range(-bound..bound),
        }
    }

    // Função para ativar o neurônio (com ReLU)
    fn activate(&self, inputs: &[f64], activation_fn: &str) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        match activation_fn {
            "linear" => sum,
            "relu" => if sum > 0.0 { sum } else { 0.0 }, // ReLU
            _ => panic!("Unsupported activation function"),
        }
    }
}

// Estrutura para representar um Perceptron Multicamadas (MLP)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MLP {
    layers: Vec<Vec<Neuron>>,
}

impl MLP {
    // Função para criar um novo MLP
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

    // Função para realizar a propagação para frente (forward propagation)
    fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![inputs.to_vec()];
        for (i, layer) in self.layers.iter().enumerate() {
            let mut new_activations = Vec::new();
            let activation_fn = if i == self.layers.len() - 1 {
                "linear"
            } else {
                "relu" // ReLU para camadas ocultas
            };
            for neuron in layer {
                new_activations.push(neuron.activate(&activations.last().unwrap(), activation_fn));
            }
            activations.push(new_activations);
        }
        activations
    }

    // Função para treinar o MLP usando retropropagação (backpropagation)
    fn train(&mut self, training_data: &[(Vec<f64>, f64)], learning_rate: f64, epochs: usize) {
        let clip_threshold = 1.0; // Limite para clipping de gradiente (ajustável)

        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (inputs, target) in training_data.iter() {
                // Propagação para frente
                let activations = self.forward(inputs);
                let output = activations.last().unwrap()[0];

                // Cálculo do erro (erro quadrático médio - MSE)
                let error = target - output;
                total_error += error * error;

                // Retropropagação
                let mut deltas = vec![vec![error]];
                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_deltas = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut delta_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            delta_sum += deltas.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        let activation = activations[l + 1][j];
                        layer_deltas.push(delta_sum * if activation > 0.0 { 1.0 } else { 0.0 }); // Delta para ReLU
                    }
                    deltas.push(layer_deltas);
                }
                deltas.reverse();

                // Atualização dos pesos e vieses (com clipping de gradiente)
                for l in 0..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        let mut gradient_norm = 0.0;
                        for k in 0..self.layers[l][j].weights.len() {
                            gradient_norm += deltas[l][j] * activations[l][k];
                        }
                        gradient_norm = gradient_norm.sqrt();

                        let scale_factor = if gradient_norm > clip_threshold {
                            clip_threshold / gradient_norm
                        } else {
                            1.0
                        };

                        for k in 0..self.layers[l][j].weights.len() {
                            self.layers[l][j].weights[k] += learning_rate * scale_factor * deltas[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * scale_factor * deltas[l][j];
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

// Função para normalizar os dados para o intervalo [0, 1]
fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let normalized: Vec<f64> = data.iter().map(|x| (x - min) / (max - min)).collect();
    (normalized, min, max)
}

fn main() {
    // Dados JSON fornecidos (exemplo simplificado)
    let json_data = r#"[{"bid":"1.0329"},{"bid":"1.0384"}]"#;

    // Parse dos dados JSON
    let parsed_data: Vec<Value> = serde_json::from_str(json_data).expect("JSON inválido");

    // Extrair os valores de 'bid'
    let bid_prices: Vec<f64> = parsed_data
        .iter()
        .map(|entry| entry["bid"].as_str().unwrap().parse::<f64>().unwrap())
        .collect();

    // Normalizar os dados
    let (normalized_prices, min_price, max_price) = normalize(&bid_prices);

    // Criar dados de treinamento (exemplo simplificado com janela de 1)
    let window_size = 1;
    let mut training_data = Vec::new();
    for i in window_size..normalized_prices.len() {
        let inputs: Vec<f64> = normalized_prices[i - window_size..i].to_vec();
        let target = normalized_prices[i];
        training_data.push((inputs, target));
    }

    // Criar e treinar a MLP (arquitetura simplificada)
    let layer_sizes = &[window_size, 2, 1]; // Ajuste a arquitetura conforme necessário
    let mut mlp = MLP::new(layer_sizes);
    mlp.train(&training_data, 0.01, 10000); // Ajuste a taxa de aprendizado e épocas

    // Prever o próximo valor (exemplo simplificado)
    let last_inputs: Vec<f64> = normalized_prices[normalized_prices.len() - window_size..].to_vec();
    let normalized_prediction = mlp.forward(&last_inputs).last().unwrap()[0];

    let denormalized_prediction = normalized_prediction * (max_price - min_price) + min_price;

    println!(
        "Previsão para o bid price: {:.4}",
        denormalized_prediction
    );
}