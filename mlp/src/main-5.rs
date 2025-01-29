use rand::Rng;
use rusqlite::{params, Connection, Result};
use serde::{Serialize, Deserialize};

// Define a estrutura de um neurônio
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize, conn: &Connection) -> Result<()> {
        for _ in 0..epochs {
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

                // Armazena os dados de treinamento no banco de dados
                let training_data_str = serde_json::to_string(&(inputs, targets)).unwrap();
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
            let (inputs, targets): (Vec<f64>, Vec<f64>) = serde_json::from_str(&data).unwrap();
            println!("Dados carregados do banco de dados: {:?}", (inputs, targets));
            // Aqui você pode usar os dados para inicializar pesos, etc.
            // Exemplo: Imprimir os dados recuperados
        } else {
            println!("Nenhum dado encontrado no banco de dados.");
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // Exemplo de uso
    let layer_sizes = &[2, 3, 1]; // Rede com 2 entradas, 3 neurônios na camada oculta e 1 saída
    let mut mlp = MLP::new(layer_sizes);

    // Conecta ao banco de dados SQLite
    let conn = Connection::open("consciencia/training_data.db")?;

    // Cria a tabela de dados de treinamento
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL
        )",
        [],
    )?;

    // Dados de treinamento (exemplo simples com XOR lógico)
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Treinamento
    mlp.train(&training_data, 0.3, 5000, &conn)?;

    // Carrega dados do banco de dados APÓS o treinamento
    mlp.load_from_db(&conn)?;

    println!("Rede treinada: {:?}", mlp);

    // Teste
    println!("Teste [0, 0]: {:?}", mlp.forward(&[0.0, 0.0]));
    println!("Teste [0, 1]: {:?}", mlp.forward(&[0.0, 1.0]));
    println!("Teste [1, 0]: {:?}", mlp.forward(&[1.0, 0.0]));
    println!("Teste [1, 1]: {:?}", mlp.forward(&[1.0, 1.0]));

    Ok(())
}


// cd mlp
// cargo run
// cargo run --bin mlp
// cargo.exe "run", "--package", "mlp", "--bin", "mlp"