use rand::Rng;
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::io;
use std::io::prelude::*;

// Defining the structure of a neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Error, Debug)]
pub enum MyError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
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
        relu(sum + self.bias)
    }
    fn activate_derivative(&self, input: f64) -> f64 {
        relu_derivative(input)
    }
}

fn normalize_data(data: Vec<f64>, min: f64, max: f64) -> Vec<f64> {
    data.iter().map(|&x| (x - min) / (max - min)).collect()
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// Defining the structure of a multi-layer perceptron (MLP)
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

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize, db_path: &str) -> Result<(), MyError> {
        // Initialize the database
        initialize_db(db_path)?;
        let conn = Connection::open(db_path)?;
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

                // Calculate the error of the output layer
                let output_activations = activations.last().unwrap();
                let output_errors: Vec<f64> = output_activations.iter().zip(targets).map(|(o, t)| (t - o)).collect();
                errors.push(output_errors);

                // Propagate errors to previous layers
                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_errors = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut error_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            error_sum += errors.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        layer_errors.push(error_sum);
                    }
                    errors.push(layer_errors);
                }
                errors.reverse();

                // Update weights and biases
                for l in 0..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        for k in 0..self.layers[l][j].weights.len() {
                            let derivative = self.layers[l][j].activate_derivative(activations[l][k]);
                            self.layers[l][j].weights[k] += learning_rate * errors[l][j] * activations[l][k] * derivative;
                        }
                        self.layers[l][j].bias += learning_rate * errors[l][j]; // Bias update
                    }
                }
            }

            print!("\rEpoch {} completed.", epoch + 1); // Print the current epoch
        }
        println!();
        // Store the trained model in the database after all training
        let trained_model_str = serde_json::to_string(&self)?;
        println!("JSON of trained model: {}", trained_model_str);
        conn.execute(
            "INSERT OR REPLACE INTO training_data (epoch, data) VALUES (?1, ?2)",
            params![epochs as i32, trained_model_str],
        )?;
        Ok(())
    }
}

fn initialize_db(db_path: &str) -> Result<()> {
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
    Ok(())
}

fn get_user_input(prompt: &str) -> Result<f64, MyError> {
    print!("{}", prompt);
    io::stdout().flush()?; // Ensures the prompt is displayed

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    input.trim().parse().map_err(|_| MyError::Io(io::Error::new(io::ErrorKind::InvalidInput, "Invalid input")))
}

fn main() -> Result<(), MyError> {
    let layer_sizes = &[3, 5, 1]; // 3 inputs, 5 hidden neurons, 1 output
    let mut mlp = MLP::new(layer_sizes);

    let db_path = "training_data.db";

    // Training data (more realistic and scaled)
    let training_data = vec![
        (vec![70.0, 30.0, 1.70], vec![24.2]), // Example: (Weight, Age, Height) -> BMI
        (vec![90.0, 40.0, 1.75], vec![29.4]),
        // ... (rest of the training data)
    ];

    mlp.train(&training_data, 0.1, 50000, db_path)?;

    println!("Network trained.");

    // Retrieve the trained model from the database
    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare("SELECT epoch, data FROM training_data")?;
    let mut rows = stmt.query([])?; 
    while let Some(row) = rows.next()? {
        let epoch: i32 = row.get(0)?;
        let data: String = row.get(1)?;
    }

    let weight = get_user_input("Digite o peso (kg): ")?;
    let age = get_user_input("Digite a idade (anos): ")?;
    let height = get_user_input("Digite a altura (metros): ")?;

    let inputs = vec![weight, age, height];
    let bmi = mlp.forward(&inputs)[0];

    let bmi_rounded = (bmi * 10.0).round() / 10.0; // Round to one decimal place

    println!("Seu IMC é: {:.1}", bmi_rounded);

    match bmi {
        bmi if bmi < 16.0 => println!("Você está em estado de magreza severa."),
        bmi if bmi < 17.0 => println!("Você está em estado de magreza."),
        bmi if bmi < 18.5 => println!("Você está abaixo do peso."),
        bmi if bmi < 25.0 => println!("Seu peso está normal."),
        bmi if bmi < 30.0 => println!("Você está com sobrepeso."),
        bmi if bmi < 35.0 => println!("Você está com obesidade grau I."),
        bmi if bmi < 40.0 => println!("Você está com obesidade grau II."),
        _ => println!("Você está com obesidade grau III."), // Obesidade mórbida
    }


    // Recomendações gerais de saúde
    println!("\nRecomendações gerais de saúde:");
    if bmi < 18.5 {
        println!("Procure um médico ou nutricionista para avaliar sua dieta e hábitos.");
    } else if bmi >= 25.0 {
        println!("Consulte um médico ou nutricionista para um plano alimentar adequado e para discutir opções de atividade física.");
    }
    println!("Mantenha uma dieta equilibrada e pratique exercícios regularmente."); // Conselho geral

    Ok(())
}

// cd mlp
// cargo run
// cargo run --bin mlp
// cargo.exe "run", "--package", "mlp", "--bin", "mlp"