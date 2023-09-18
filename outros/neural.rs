use std::collections::HashMap;

struct NeuralNetwork {
    weights: HashMap<String, f32>,
    biases: HashMap<String, f32>,
}

impl NeuralNetwork {
    fn new() -> Self {
        NeuralNetwork {
            weights: HashMap::new(),
            biases: HashMap::new(),
        }
    }

    fn train(&mut self, inputs: &[f32], outputs: &[f32]) {
        // TODO: Implement training algorithm.
    }

    fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        // TODO: Implement prediction algorithm.
        let mut output = Vec::new();
        for input in inputs {
            let mut sum = 0.0;
            for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
                sum += weight * input + bias;
            }
            output.push(sum);
        }
        output
    }
}

fn main() {
    let mut network = NeuralNetwork::new();

    // Train the network.
    network.train(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    network.train(&[2.5, 2.9, 3.0], &[4.0, 8.0, 7.0]);
    network.train(&[2.0, 3.0, 4.0], &[5.0, 3.0, 5.0]);
    network.train(&[5.0, 2.0, 3.0], &[9.0, 5.0, 6.0]);

    // Predict the output for a new input.
    let output = network.predict(&[1.0, 2.0, 3.0]);

    // Print the output.
    println!("{:?}", output);
}