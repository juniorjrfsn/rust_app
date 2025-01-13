use rand::Rng;
use rusqlite::{Connection, Result };
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

#[derive(Debug)]
struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Perceptron {
    fn new(input_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        Perceptron {
            weights: (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: rng.gen_range(-1.0..1.0),
            learning_rate,
        }
    }

    fn predict(&self, inputs: &[f64]) -> i32 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        if sum > 0.0 {
            1
        } else {
            0
        }
    }

    fn train(&mut self, inputs: &[f64], target: i32) {
        let prediction = self.predict(inputs);
        let error = target - prediction;

        for i in 0..inputs.len() {
            self.weights[i] += self.learning_rate * error as f64 * inputs[i];
        }
        self.bias += self.learning_rate * error as f64;
    }

    fn save_to_db(&self, conn: &Connection, table_name: &str) -> Result<()> {
        let weights_str = self.weights.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(",");

        conn.execute(
            &format!("CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, weights TEXT, bias REAL, learning_rate REAL)", table_name),
            [],
        )?;

        conn.execute(
            &format!("INSERT INTO {} (weights, bias, learning_rate) VALUES (?, ?, ?)", table_name),
            (&weights_str, self.bias, self.learning_rate),
        )?;

        Ok(())
    }

    fn load_from_db(conn: &Connection, table_name: &str, id: i32) -> Result<Option<Self>> {
        let mut stmt = conn.prepare(&format!("SELECT weights, bias, learning_rate FROM {} WHERE id = ?", table_name))?;
        let mut rows = stmt.query(rusqlite::params![id])?;

        if let Some(row) = rows.next()? {
            let weights_str: String = row.get(0)?;
            let weights: Vec<f64> = weights_str.split(",").map(|s| s.parse().unwrap()).collect();
            let bias: f64 = row.get(1)?;
            let learning_rate: f64 = row.get(2)?;

            Ok(Some(Perceptron {
                weights,
                bias,
                learning_rate,
            }))
        } else {
            Ok(None)
        }
    }

    fn update_in_db(&self, conn: &Connection, table_name: &str, id: i32) -> Result<()> {
        let weights_str = self.weights.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(",");
        conn.execute(
            &format!("UPDATE {} SET weights = ?, bias = ?, learning_rate = ? WHERE id = ?", table_name),
            (&weights_str, self.bias, self.learning_rate, id),
        )?;
        Ok(())
    }
}
fn read_data_from_file<P: AsRef<Path>>(filename: P) -> Result<Vec<(Vec<f64>, i32)>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.trim().split(';').collect(); // Usando ';' como separador

        if parts.len() != 2 {
            eprintln!("Linha formatada incorretamente: {}", line);
            continue; // Ignora linhas mal formatadas
        }

        let inputs_str = parts[0];
        let target_str = parts[1];

        let inputs: Result<Vec<f64>, _> = inputs_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect();

        let target: Result<i32, _> = target_str.trim().parse();

        match (inputs, target) {
            (Ok(inputs), Ok(target)) => data.push((inputs, target)),
            (Err(e), _) => eprintln!("Erro ao analisar entradas: {}", e),
            (_, Err(e)) => eprintln!("Erro ao analisar target: {}", e),
        }
    }

    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    const INPUT_SIZE: usize = 784; // Ajuste conforme necessário
    let learning_rate = 0.01;
    let mut perceptron = Perceptron::new(INPUT_SIZE, learning_rate);

    // Ler dados de treinamento do arquivo
    let training_data = match read_data_from_file("training_data.txt") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Erro ao ler dados de treinamento: {}", e);
            return Ok(()); // Sai do programa graciosamente
        }
    };

    if training_data.is_empty() {
        eprintln!("Nenhum dado de treinamento lido do arquivo.");
        return Ok(());
    }

    let epochs = 100;
    for epoch in 0..epochs {
        for (inputs, target) in &training_data {
            perceptron.train(inputs, *target);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}", epoch);
            let mut correct_predictions = 0;
            for (inputs, target) in &training_data {
                if perceptron.predict(inputs) == *target {
                    correct_predictions += 1;
                }
            }
            println!(
                "Acurácia: {}/{} = {}",
                correct_predictions,
                training_data.len(),
                correct_predictions as f32 / training_data.len() as f32
            );
        }
    }
    /*
    const INPUT_SIZE: usize = 784;
    let learning_rate = 0.01;
    let mut perceptron = Perceptron::new(INPUT_SIZE, learning_rate);

    let all_zeros = vec![0.0; INPUT_SIZE];
    let all_ones = vec![1.0; INPUT_SIZE];
    let half_gray = vec![0.5; INPUT_SIZE];
    let dark_gray = vec![0.2; INPUT_SIZE];
    let light_gray = vec![0.7; INPUT_SIZE];
    let almost_all_ones = vec![0.99; INPUT_SIZE];
    let almost_all_zeros = vec![0.01; INPUT_SIZE];

    let training_data: Vec<(&[f64], i32)> = vec![
        (&all_zeros, 0),
        (&all_ones, 1),
        (&half_gray, 1),
        (&dark_gray, 0),
        (&light_gray, 1),
        (&almost_all_ones, 1),
        (&almost_all_zeros, 0),
    ];

    let epochs = 100;
    for epoch in 0..epochs {
        for (inputs, target) in &training_data {
            perceptron.train(inputs, *target);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}", epoch);
            let mut correct_predictions = 0;
            for (inputs, target) in &training_data {
                if perceptron.predict(inputs) == *target {
                    correct_predictions += 1;
                }
            }
            println!(
                "Accuracy: {}/{} = {}",
                correct_predictions,
                training_data.len(),
                correct_predictions as f32 / training_data.len() as f32
            );
        }
    }
    */

    // Salvar no banco de dados
    let conn = Connection::open("perceptron.db")?;
    perceptron.save_to_db(&conn, "my_perceptron")?;
    println!("Perceptron salvo no banco de dados.");

    // Carregar do banco de dados (exemplo)
    if let Ok(Some(loaded_perceptron)) = Perceptron::load_from_db(&conn, "my_perceptron", 1) {
        println!("Perceptron carregado do banco de dados: {:?}", loaded_perceptron);
        // Testar o perceptron carregado
        let test_all_zeros = vec![0.0; INPUT_SIZE];
        println!("Predição do perceptron carregado para zeros: {}", loaded_perceptron.predict(&test_all_zeros));
    }
    else {
        println!("Falha ao carregar o perceptron do banco de dados.");
    }

    Ok(())
}