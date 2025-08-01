// File: cotacaomlp/src/mlp/mlp_cotacao.rs 
pub mod rna { // módulo rna : machine learning MLP (Multi-Layer Perceptron)
    use rand::Rng;
    use serde::{Serialize, Deserialize};
    use bincode;
    use crate::mlp::mlp_cotacao::rna;
    
    // Função para aplicar a ativação escolhida
    fn apply_activation(x: f64, activation: &str) -> f64 {
        match activation {
            "relu" => relu(x),
            "tanh" => tanh(x),
            "sigmoid" => sigmoid(x),
            _ => x, // Linear (sem ativação)
        }
    }
    
    // Função para calcular a derivada da ativação escolhida
    fn activation_derivative(x: f64, activation: &str) -> f64 {
        match activation {
            "relu" => relu_derivative(x),
            "tanh" => tanh_derivative(x),
            "sigmoid" => sigmoid_derivative(x),
            _ => 1.0, // Derivada da função linear
        }
    }
    
    // Função de ativação Tanh
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }
    
    // Derivada da função de ativação Tanh
    fn tanh_derivative(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
    
    // Função de ativação Sigmoid
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // Derivada da função de ativação Sigmoid
    fn sigmoid_derivative(x: f64) -> f64 {
        let s = sigmoid(x);
        s * (1.0 - s)
    }
    
    // Função de ativação ReLU
    fn relu(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }
    
    // Derivada da função de ativação ReLU
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    
    // Estrutura para representar uma camada densa
    #[derive(Serialize, Deserialize)]
    pub struct DenseLayer {
        pub weights: Vec<Vec<f64>>,
        pub biases: Vec<f64>,
    }
    
    impl DenseLayer {
        // Cria uma nova camada densa com inicialização Xavier
        pub fn new(input_size: usize, output_size: usize) -> Self {
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let mut rng = rand::rng();
            let weights = (0..output_size)
                .map(|_| (0..input_size).map(|_| rng.gen_range(-scale..scale)).collect())
                .collect();
            let biases = vec![0.0; output_size];
            DenseLayer { weights, biases }
        }
    
        // Propagação para frente (forward pass)
        pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
            self.weights
                .iter()
                .zip(&self.biases)
                .map(|(weights, bias)| {
                    inputs
                        .iter()
                        .zip(weights)
                        .map(|(x, w)| x * w)
                        .sum::<f64>()
                        + bias
                })
                .collect()
        }
    
        // Atualiza os pesos usando gradientes e taxa de aprendizado
        pub fn update_weights(&mut self, gradients: &[Vec<f64>], learning_rate: f64) {
            for (i, row) in self.weights.iter_mut().enumerate() {
                for (j, weight) in row.iter_mut().enumerate() {
                    *weight -= learning_rate * gradients[i][j];
                }
            }
        }
    }
    
    // Estrutura para representar o modelo MLP
    #[derive(Serialize, Deserialize)]
    pub struct MLP {
        pub layers: Vec<DenseLayer>,
    }
    
    impl MLP {
        // Cria um novo modelo MLP com tamanhos de camadas especificados
        pub fn new(layer_sizes: &[usize]) -> Self {
            let layers = layer_sizes
                .windows(2)
                .map(|sizes| DenseLayer::new(sizes[0], sizes[1]))
                .collect();
            MLP { layers }
        }
    
        // Propagação para frente (forward pass)
        pub fn forward(&self, inputs: &[f64], activation: &str) -> Vec<f64> {
            let mut output = inputs.to_vec();
            for layer in &self.layers {
                output = layer.forward(&output);
                output = output.iter().map(|x| apply_activation(*x, activation)).collect();
            }
            output
        }
    
        // Treina o modelo usando gradiente descendente
        pub fn train(
            &mut self,
            inputs: &[Vec<f64>],
            labels: &[f64],
            epochs: usize,
            learning_rate: f64,
            activation: &str,
        ) {
            for epoch in 0..epochs {
                let mut total_loss = 0.0;
                for (input, label) in inputs.iter().zip(labels) {
                    // Forward pass
                    let mut outputs = vec![input.clone()];
                    for layer in &self.layers {
                        let output = layer.forward(outputs.last().unwrap());
                        outputs.push(output.iter().map(|x| apply_activation(*x, activation)).collect());
                    }
    
                    // Calcula a perda (MSE)
                    let prediction = outputs.last().unwrap()[0];
                    let loss = (prediction - label).powi(2);
                    total_loss += loss;
    
                    // Backward pass
                    let mut delta = 2.0 * (prediction - label);
                    for i in (0..self.layers.len()).rev() {
                        let output = &outputs[i + 1];
                        let input = &outputs[i];
                        let gradients: Vec<Vec<f64>> = self.layers[i]
                            .weights
                            .iter()
                            .enumerate()
                            .map(|(j, weights)| {
                                weights
                                    .iter()
                                    .enumerate()
                                    .map(|(k, _)| {
                                        delta * activation_derivative(output[j], activation) * input[k]
                                    })
                                    .collect()
                            })
                            .collect();
                        self.layers[i].update_weights(&gradients, learning_rate);
                        delta = self.layers[i]
                            .weights
                            .iter()
                            .map(|weights| {
                                weights
                                    .iter()
                                    .zip(output.iter())
                                    .map(|(w, o)| w * activation_derivative(*o, activation))
                                    .sum::<f64>()
                            })
                            .sum();
                    }
                }
                println!("Epoch: {}, Loss: {:.4}", epoch + 1, total_loss / inputs.len() as f64);
            }
        }
    
        // Serializa o modelo para bytes
        pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let serialized = bincode::serialize(self)?;
            Ok(serialized)
        }
    
        // Desserializa o modelo a partir de bytes
        pub fn deserialize(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
            let deserialized = bincode::deserialize(data)?;
            Ok(deserialized)
        }
    }

}