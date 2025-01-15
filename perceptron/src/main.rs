use rand::Rng;

#[derive(Debug)]
struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    bias_output: f64,
    learning_rate: f64,
}

impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights_input_hidden = (0..input_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();

        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        let bias_hidden = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = rng.gen_range(-1.0..1.0);

        MLP {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            learning_rate,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn forward(&self, inputs: &Vec<f64>) -> f64 {
        let hidden_outputs: Vec<f64> = (0..self.hidden_size)
            .map(|j| {
                let u = inputs
                    .iter()
                    .enumerate()
                    .fold(self.bias_hidden[j], |acc, (i, &x)| acc + x * self.weights_input_hidden[i][j]);
                Self::sigmoid(u)
            })
            .collect();

        let output = hidden_outputs.iter().enumerate().fold(self.bias_output,|acc, (i, &x)| acc + x * self.weights_hidden_output[i][0]);
        Self::sigmoid(output)
    }

    fn train(&mut self, inputs: &Vec<f64>, target: f64) {
        // Forward pass
        let hidden_outputs: Vec<f64> = (0..self.hidden_size)
            .map(|j| {
                let u = inputs
                    .iter()
                    .enumerate()
                    .fold(self.bias_hidden[j], |acc, (i, &x)| acc + x * self.weights_input_hidden[i][j]);
                Self::sigmoid(u)
            })
            .collect();

        let output = hidden_outputs.iter().enumerate().fold(self.bias_output,|acc, (i, &x)| acc + x * self.weights_hidden_output[i][0]);
        let output = Self::sigmoid(output);

        // Backpropagation
        let output_error = (target - output) * Self::sigmoid_derivative(output);

        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|j| {
                output_error * self.weights_hidden_output[j][0] * Self::sigmoid_derivative(hidden_outputs[j])
            })
            .collect();

        // Update weights and biases
        for j in 0..self.hidden_size {
            for i in 0..self.input_size {
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_errors[j] * inputs[i];
            }
            self.bias_hidden[j] += self.learning_rate * hidden_errors[j];
        }

        for j in 0..self.hidden_size{
            self.weights_hidden_output[j][0] += self.learning_rate * output_error * hidden_outputs[j];
        }
        self.bias_output += self.learning_rate * output_error;
    }
}

fn main() {
    let mut mlp = MLP::new(2, 2, 1, 0.5); // 2 entradas, 2 neurônios ocultos, 1 saída, taxa de aprendizado 0.5
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    for _ in 0..10000 { // Treina por 10000 épocas
        for (inputs, target) in &training_data {
            mlp.train(inputs, *target);
        }
    }

    // Teste
    for (inputs, target) in &training_data {
        let output = mlp.forward(inputs);
        let threshold = 0.5;
        let formatted_output = if output >= threshold { 1.0 } else { 0.0 };
        let classification   = if output >= threshold { 1.0 } else { 0.0 };

        println!("Inputs: {:?}, Target: {}, Output: {:.4} (Output XOR: {})", inputs, target, output, classification);
    }
}