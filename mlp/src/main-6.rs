use rand::Rng;
use rand_distr::{Distribution, Normal};
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        Neuron {
            weights: (0..num_inputs).map(|_| normal.sample(&mut rng)).collect(),
            bias: normal.sample(&mut rng),
        }
    }

    fn activate(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
        1.0 / (1.0 + (-sum - self.bias).exp())
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

    fn forward(&self, inputs: &[f64]) -> f64 {
        let mut activations = inputs.to_vec();
        for layer in &self.layers {
            let mut new_activations = Vec::new();
            for neuron in layer {
                new_activations.push(neuron.activate(&activations));
            }
            activations = new_activations;
        }
        activations[0]
    }

    fn train(&mut self, training_data: &[(Vec<f64>, f64)], learning_rate: f64, epochs: usize, conn: &Connection) -> Result<()> {
         for _ in 0..epochs {
            for (inputs, target) in training_data.iter() {
                let prediction = self.forward(inputs);
                let error = target - prediction;

                let mut activations = vec![inputs.clone()];
                for layer in &self.layers {
                    let new_activations: Vec<f64> = layer.iter().map(|neuron| neuron.activate(activations.last().unwrap())).collect();
                    activations.push(new_activations);
                }
                let mut deltas = Vec::new();

                let output_delta = error * prediction * (1.0 - prediction);
                deltas.push(vec![output_delta]);

                for l in (0..self.layers.len() - 1).rev() {
                    let mut layer_deltas = Vec::new();
                    for j in 0..self.layers[l].len() {
                        let mut delta_sum = 0.0;
                        for k in 0..self.layers[l + 1].len() {
                            delta_sum += deltas.last().unwrap()[k] * self.layers[l + 1][k].weights[j];
                        }
                        let activation = activations[l + 1][j];
                        layer_deltas.push(delta_sum * activation * (1.0 - activation));
                    }
                    deltas.push(layer_deltas);
                }
                deltas.reverse();

                for l in 0..self.layers.len() {
                    for j in 0..self.layers[l].len() {
                        for k in 0..self.layers[l][j].weights.len() {
                            self.layers[l][j].weights[k] += learning_rate * deltas[l][j] * activations[l][k];
                        }
                        self.layers[l][j].bias += learning_rate * deltas[l][j];
                    }
                }
                let training_data_str = serde_json::to_string(&(inputs, target)).unwrap();
                conn.execute(
                    "INSERT INTO training_data (data) VALUES (?1)",
                    params![training_data_str],
                )?;
            }
        }
        Ok(())
    }

    fn load_from_db(&mut self, conn: &Connection) -> Result<()> {
         let mut stmt = conn.prepare("SELECT data FROM training_data ORDER BY id DESC LIMIT 1")?;
        let mut rows = stmt.query([])?;

        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            let _: (Vec<f64>, f64) = serde_json::from_str(&data).unwrap();
            println!("Dados carregados do banco de dados.");
        } else {
            println!("Nenhum dado encontrado no banco de dados.");
        }
        Ok(())
    }
}

fn interpret_bmi(bmi: f64) -> String {
    if bmi < 18.5 {
        "Abaixo do peso".to_string()
    } else if bmi < 25.0 {
        "Peso normal".to_string()
    } else if bmi < 30.0 {
        "Sobrepeso".to_string()
    } else {
        "Obesidade".to_string()
    }
}

fn main() -> std::result::Result<(), Box<dyn Error>>{
    let layer_sizes = &[2, 3, 1];
    let mut mlp = MLP::new(layer_sizes);

    let conn = Connection::open("consciencia/training_data.db")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL
        )",
        [],
    )?;

    // Dados de treinamento para IMC (peso em kg, altura em metros, IMC)
    let training_data = vec![
        (vec![70.0, 1.75], 22.86), // Peso normal
        (vec![90.0, 1.80], 27.78), // Sobrepeso
        (vec![60.0, 1.60], 23.44), // Peso normal
        (vec![110.0, 1.70], 38.06),// Obesidade
        (vec![50.0, 1.70], 17.30), // Abaixo do peso
        (vec![85.0, 1.75], 27.76), //Sobrepeso
        (vec![120.0, 1.85],35.15), //Obesidade
        (vec![60.0, 1.80],18.52), //Peso normal
    ];

    mlp.train(&training_data, 0.1, 5000, &conn)?;
    mlp.load_from_db(&conn)?;

    // Teste com novos dados
    let peso = 75.0;
    let altura = 1.78;
    let input_teste = vec![peso, altura];
    let imc_calculado = mlp.forward(&input_teste);

    println!("Peso: {} kg, Altura: {} m", peso, altura);
    println!("IMC Calculado: {:.2}", imc_calculado);
    println!("Interpretação: {}", interpret_bmi(imc_calculado));

    Ok(())
}

// cd mlp
// cargo run
// cargo run --bin mlp
// cargo.exe "run", "--package", "mlp", "--bin", "mlp"