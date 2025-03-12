use ndarray::{Array1, Array2, Array, arr1};
use std::collections::HashMap;
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, Write, Read};
use std::path::Path;

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

#[derive(Serialize, Deserialize)]
pub struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    bias_hidden: Array1<f64>,
    bias_output: Array1<f64>,
    learning_rate: f64,
}

impl MLP {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, (1.0 / input_size as f64).sqrt()).unwrap();
        let weights_input_hidden = Array::from_shape_fn(
            (input_size, hidden_size),
            |_| normal.sample(&mut rng)
        );

        let normal_hidden = Normal::new(0.0, (1.0 / hidden_size as f64).sqrt()).unwrap();
        let weights_hidden_output = Array::from_shape_fn(
            (hidden_size, output_size),
            |_| normal_hidden.sample(&mut rng)
        );

        MLP {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden: Array1::zeros(hidden_size),
            bias_output: Array1::zeros(output_size),
            learning_rate,
        }
    }

    pub fn forward(&self, inputs: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden_inputs = self.weights_input_hidden.t().dot(inputs) + &self.bias_hidden;
        let hidden_outputs = hidden_inputs.mapv(relu);
        let output_inputs = self.weights_hidden_output.t().dot(&hidden_outputs) + &self.bias_output;
        let output = softmax(&output_inputs);
        (hidden_outputs, output)
    }

    pub fn train(&mut self, inputs: &Array1<f64>, target: &Array1<f64>) {
        let (hidden_outputs, output) = self.forward(inputs);
        let output_errors = &output - target;

        let relu_deriv = hidden_outputs.mapv(relu_derivative);
        let hidden_errors = self.weights_hidden_output.dot(&output_errors) * relu_deriv;

        let outer_product = hidden_outputs.to_owned().into_shape((hidden_outputs.len(), 1)).unwrap() *
            output_errors.to_owned().into_shape((1, output_errors.len())).unwrap();
        self.weights_hidden_output -= &(outer_product * self.learning_rate);
        self.bias_output -= &(output_errors * self.learning_rate);

        let input_outer = inputs.to_owned().into_shape((inputs.len(), 1)).unwrap() *
            hidden_errors.to_owned().into_shape((1, hidden_errors.len())).unwrap();
        self.weights_input_hidden -= &(input_outer * self.learning_rate);
        self.bias_hidden -= &(hidden_errors * self.learning_rate);
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let mlp = serde_json::from_str(&contents)?;
        Ok(mlp)
    }
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let exp_x = x.mapv(|v| v.exp());
    let sum_exp = exp_x.sum();
    exp_x / sum_exp
}

fn train_model() -> io::Result<()> {
    let training_data: Vec<([u8; 3], &str)> = vec![
        ([255, 0, 127], "Rose"),
        ([127, 0, 0], "Vermelho"),
        ([255, 0, 0], "Vermelho"),
        ([255, 127, 0], "Laranja"),
        ([127, 127, 0], "Amarelo"),
        ([255, 255, 0], "Amarelo"),
        ([127, 255, 0], "Primavera"),
        ([0, 127, 0], "Verde"),
        ([0, 255, 0], "Verde"),
        ([0, 255, 127], "Turquesa"),
        ([0, 127, 127], "Ciano"),
        ([0, 255, 255], "Ciano"),
        ([0, 127, 255], "Cobalto"),
        ([0, 0, 127], "Azul"),
        ([0, 0, 255], "Azul"),
        ([127, 0, 255], "Violeta"),
        ([127, 0, 127], "Magenta"),
        ([255, 0, 255], "Magenta"),
        ([0, 0, 0], "Preto"),
        ([127, 127, 127], "Cinza"),
        ([255, 255, 255], "Branco"),
    ];

    let colors: Vec<&str> = training_data.iter()
        .map(|(_, c)| *c)
        .unique()
        .sorted()
        .collect();
    let color_idx: HashMap<&str, usize> = colors.iter().enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let normalized_data: Vec<(Array1<f64>, Array1<f64>)> = training_data.iter()
        .map(|&(rgb, color)| {
            let inputs = arr1(&[
                rgb[0] as f64 / 255.0,
                rgb[1] as f64 / 255.0,
                rgb[2] as f64 / 255.0,
            ]);
            let mut target = Array1::zeros(colors.len());
            target[color_idx[color]] = 1.0;
            (inputs, target)
        })
        .collect();

    let model_path = "dados/color_mlp.json";

    // Create directory if it doesn't exist
    if let Some(parent) = Path::new(model_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Training new model...");
    let mut mlp = MLP::new(3, 32, colors.len(), 0.01);
    let epochs = 10000;

    for epoch in 0..epochs {
        for (inputs, target) in &normalized_data {
            mlp.train(inputs, target);
        }

        if epoch % 1000 == 0 {
            let loss: f64 = normalized_data.iter()
                .map(|(i, t)| {
                    let (_, o) = mlp.forward(i);
                    let target_class = t.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    -o[target_class].ln()
                })
                .sum();
            println!("Epoch {:5} | Loss: {:.4}", epoch, loss);
        }
    }

    mlp.save(model_path)?;
    println!("Model saved to {}", model_path);
    Ok(())
}

fn test_model() -> io::Result<()> {
    let model_path = "dados/color_mlp.json";

    // Load color classes (need to match training data)
    let color_classes: Vec<&str> = vec![
        "Amarelo", "Azul", "Branco", "Ciano", "Cinza", "Cobalto",
        "Laranja", "Magenta", "Preto", "Primavera", "Rose",
        "Turquesa", "Verde", "Vermelho", "Violeta"
    ].into_iter().sorted().collect();

    // Load the trained model
    let mlp = MLP::load(model_path)?;
    println!("Loaded model from {}", model_path);

    // Test data
    let test_data = vec![
        [200, 0, 70],
        [240, 100, 240],
    ];

    println!("\n================ TESTE ===============");
    for rgb in test_data {
        let inputs = arr1(&[
            rgb[0] as f64 / 255.0,
            rgb[1] as f64 / 255.0,
            rgb[2] as f64 / 255.0,
        ]);

        let (_, output) = mlp.forward(&inputs);
        let max_idx = output.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let predicted = color_classes[max_idx];
        let confidence = output[max_idx] * 100.0;

        println!(
            "Cor RGB: {:?} | Previsão: {:<10} | Confiança: {:.1}%",
            rgb, predicted, confidence
        );
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "treino" => train_model(),
            "reconhecer" => test_model(),
            _ => {
                println!("Usage: {} [treino|reconhecer]", args[0]);
                Ok(())
            }
        }
    } else {
        println!("Usage: {} [treino|reconhecer]", args[0]);
        Ok(())
    }
}


 /*

 cd reconhececor
cargo run -- treino

 ...

cd reconhececor
cargo run -- reconhecer

*/
