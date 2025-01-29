use rand::Rng;
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::io;
use std::io::prelude::*;

// Define a estrutura de um neurônio
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
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
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

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize, db_path: &str) -> Result<(), MyError> {
        // Inicializa o banco de dados
        initialize_db(db_path)?;
        let conn = Connection::open(db_path)?;
        let mut training_data_str:String = "".to_string();
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
                let output_errors: Vec<f64> = output_activations.iter().zip(targets).map(|(o, t)| (t - o)).collect();
                errors.push(output_errors);

                // Propaga os erros para as camadas anteriores
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

                // Atualiza os pesos e biases
                for l in 0..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        for k in 0..self.layers[l][j].weights.len() {
                            self.layers[l][j].weights[k] += learning_rate * errors[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * errors[l][j];
                    }
                }
                // training_data_str = serde_json::to_string(&self)?;
            }

            println!("Epoch {} completed.", epoch + 1);
        }

        // Armazena o modelo treinado no banco de dados após todo o treinamento
        // let trained_model_str = serde_json::to_string(&self)?;
        let trained_model_str = serde_json::to_string(&self)?;
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
    io::stdout().flush()?; // Garante que o prompt seja exibido

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    input.trim().parse().map_err(|_| MyError::Io(io::Error::new(io::ErrorKind::InvalidInput, "Entrada inválida")))
}

fn main() -> Result<(), MyError> {
    let layer_sizes = &[3, 5, 1]; // 3 inputs, 5 hidden neurons, 1 output
    let mut mlp = MLP::new(layer_sizes);

    let db_path = "training_data.db";

    // Training data (more realistic and scaled)
    let training_data = vec![
        (vec![70.0, 30.0, 1.70], vec![24.2]), // Example: (Weight, Age, Height) -> BMI
        (vec![90.0, 40.0, 1.75], vec![29.4]),
        (vec![60.0, 25.0, 1.60], vec![23.4]),
        (vec![75.0, 35.0, 1.65], vec![27.5]),
        (vec![80.0, 45.0, 1.80], vec![24.7]),
        (vec![65.0, 30.0, 1.70], vec![22.5]),
        (vec![85.0, 50.0, 1.75], vec![27.7]),
        (vec![70.0, 40.0, 1.60], vec![27.3]),
        (vec![95.0, 55.0, 1.85], vec![27.8]),
        (vec![60.0, 20.0, 1.65], vec![22.0]),
        (vec![100.0, 60.0, 1.90], vec![27.7]),
        (vec![75.0, 25.0, 1.70], vec![25.9]),
        (vec![80.0, 35.0, 1.75], vec![26.1]),
        (vec![100.0, 34.0, 1.80], vec![30.9]),
        (vec![55.0, 28.0, 1.55], vec![24.2]), // New data point
        (vec![92.0, 42.0, 1.82], vec![27.8]), // New data point
        (vec![68.0, 32.0, 1.68], vec![24.0]),  // New data point
        (vec![78.0, 38.0, 1.72], vec![26.3]),  // New data point
        (vec![88.0, 48.0, 1.78], vec![27.7]),   // New data point
        (vec![72.0, 36.0, 1.65], vec![26.5]),   // New data point
    ];

    mlp.train(&training_data, 0.3, 50000, db_path)?;

    println!("Rede treinada.");

    let peso = get_user_input("Digite o peso (kg): ")?;
    let idade = get_user_input("Digite a idade (anos): ")?;
    let altura = get_user_input("Digite a altura (metros): ")?;

    let inputs = vec![peso, idade, altura];
    let imc = mlp.forward(&inputs)[0];

    println!("Seu IMC é: {:.2}", imc);

    // Improved Recommendations (More Detailed)
    let imc_rounded = (imc * 10.0).round() / 10.0; // Round to one decimal place for display

    println!("Seu IMC é: {:.1}", imc_rounded); // Display rounded IMC

    match imc {
        imc if imc < 16.0 => println!("Você está em estado de magreza severa."),
        imc if imc < 17.0 => println!("Você está em estado de magreza."),
        imc if imc < 18.5 => println!("Você está abaixo do peso."),
        imc if imc < 25.0 => println!("Seu peso está normal."),
        imc if imc < 30.0 => println!("Você está com sobrepeso."),
        imc if imc < 35.0 => println!("Você está com obesidade grau I."),
        imc if imc < 40.0 => println!("Você está com obesidade grau II."),
        _ => println!("Você está com obesidade grau III."), // Morbid obesity
    }

    // General Health Recommendations (Placeholder - Consult a professional)
    println!("\nRecomendações gerais de saúde:");
    if imc < 18.5 {
        println!("Procure um médico ou nutricionista para avaliar sua dieta e hábitos.");
    } else if imc >= 25.0 {
        println!("Consulte um médico ou nutricionista para um plano alimentar adequado e para discutir opções de atividade física.");
    }
    println!("Mantenha uma dieta equilibrada e pratique exercícios regularmente."); // General advice

    Ok(())
}


// cd mlp
// cargo run
// cargo run --bin mlp
// cargo.exe "run", "--package", "mlp", "--bin", "mlp"