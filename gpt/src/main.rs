use rand::Rng; // Importa o gerador de números aleatórios
use serde::{Deserialize, Serialize}; // Importa as traits para serialização e deserialização

// Estrutura para representar um neurônio
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>, // Vetor de pesos (um para cada entrada)
    bias: f64,          // Viés do neurônio
}

impl Neuron {
    // Função para criar um novo neurônio
    fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng(); // Cria um gerador de números aleatórios
        let bound = (6.0 / (num_inputs + 1) as f64).sqrt(); // Calcula o limite para inicialização de Xavier
        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-bound..bound)).collect(), // Inicializa os pesos com valores aleatórios dentro do limite
            bias: rng.gen_range(-bound..bound), // Inicializa o viés com um valor aleatório dentro do limite
        }
    }

    // Função para ativar o neurônio
    fn activate(&self, inputs: &[f64], activation_fn: &str) -> f64 {
        let sum: f64 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>() + self.bias; // Calcula a soma ponderada das entradas e adiciona o viés
        match activation_fn {
            "linear" => sum, // Função de ativação linear (para a camada de saída em problemas de regressão)
            "tanh" => sum.tanh(), // Função de ativação tanh (para camadas ocultas)
            _ => panic!("Unsupported activation function"), // Lança um erro se a função de ativação não for suportada
        }
    }
}

// Estrutura para representar um Perceptron Multicamadas (MLP)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MLP {
    layers: Vec<Vec<Neuron>>, // Vetor de camadas, onde cada camada é um vetor de neurônios
}

impl MLP {
    // Função para criar um novo MLP
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new(); // Vetor para armazenar as camadas
        for i in 0..layer_sizes.len() - 1 { // Itera sobre as camadas (exceto a última)
            let num_neurons = layer_sizes[i + 1]; // Número de neurônios na camada atual
            let num_inputs = layer_sizes[i]; // Número de entradas para a camada atual (igual ao número de neurônios na camada anterior)
            let layer = (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect(); // Cria uma nova camada com os neurônios
            layers.push(layer); // Adiciona a camada ao MLP
        }
        MLP { layers }
    }

    // Função para realizar a propagação para frente (forward propagation)
    fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![inputs.to_vec()]; // Vetor para armazenar as ativações de cada camada, inicializado com as entradas
        for (i, layer) in self.layers.iter().enumerate() { // Itera sobre as camadas
            let mut new_activations = Vec::new(); // Vetor para armazenar as ativações da camada atual
            let activation_fn = if i == self.layers.len() - 1 {
                "linear" // Função de ativação linear para a camada de saída
            } else {
                "tanh" // Função de ativação tanh para as camadas ocultas
            };
            for neuron in layer { // Itera sobre os neurônios da camada
                new_activations.push(neuron.activate(&activations.last().unwrap(), activation_fn)); // Calcula a ativação do neurônio e adiciona ao vetor
            }
            activations.push(new_activations); // Adiciona as ativações da camada atual ao vetor
        }
        activations // Retorna o vetor com as ativações de todas as camadas
    }

    // Função para treinar o MLP usando retropropagação (backpropagation)
    fn train(&mut self, training_data: &[(Vec<f64>, f64)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs { // Itera sobre as épocas de treinamento
            let mut total_error = 0.0; // Variável para armazenar o erro total da época
            for (inputs, target) in training_data.iter() { // Itera sobre os dados de treinamento
                // Propagação para frente
                let activations = self.forward(inputs); // Calcula as ativações de todas as camadas
                let output = activations.last().unwrap()[0]; // Obtém a saída do MLP

                // Cálculo do erro (erro quadrático médio - MSE)
                let error = target - output; // Calcula o erro para a saída atual
                total_error += error * error; // Acumula o erro total

                // Retropropagação
                let mut deltas = vec![vec![error]]; // Inicializa os deltas para a camada de saída
                for l in (0..self.layers.len() - 1).rev() { // Itera sobre as camadas (da penúltima para a primeira)
                    let mut layer_deltas = Vec::new(); // Vetor para armazenar os deltas da camada atual
                    for j in 0..self.layers[l].len() { // Itera sobre os neurônios da camada atual
                        let mut delta_sum = 0.0; // Variável para acumular a soma dos deltas ponderados da camada seguinte
                        for k in 0..self.layers[l + 1].len() { // Itera sobre os neurônios da camada seguinte
                            delta_sum += deltas.last().unwrap()[k] * self.layers[l + 1][k].weights[j]; // Calcula a soma ponderada dos deltas da camada seguinte
                        }
                        let activation = activations[l + 1][j]; // Obtém a ativação do neurônio atual
                        layer_deltas.push(delta_sum * (1.0 - activation * activation)); // Calcula o delta do neurônio atual (derivada da função tanh)
                    }
                    deltas.push(layer_deltas); // Adiciona os deltas da camada atual ao vetor
                }
                deltas.reverse(); // Inverte o vetor de deltas para ficar na ordem correta

                // Atualização dos pesos e vieses
                for l in 0..self.layers.len() { // Itera sobre as camadas
                    for j in 0..self.layers[l].len() { // Itera sobre os neurônios da camada
                        for k in 0..self.layers[l][j].weights.len() { // Itera sobre os pesos do neurônio
                            self.layers[l][j].weights[k] += learning_rate * deltas[l][j] * activations[l][k]; // Atualiza o peso
                        }
                        self.layers[l][j].bias += learning_rate * deltas[l][j]; // Atualiza o viés
                    }
                }
            }
            let avg_error = total_error / training_data.len() as f64; // Calcula o erro médio da época
            if epoch % 1000 == 0 || epoch == epochs - 1 { // Imprime o erro a cada 1000 épocas ou na última época
                println!("Epoch {}: Average Error = {:.6}", epoch + 1, avg_error);
            }
        }
    }
}

// Função para normalizar os dados para o intervalo [0, 1]
fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min); // Encontra o valor mínimo nos dados
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Encontra o valor máximo nos dados
    let normalized: Vec<f64> = data.iter().map(|x| (x - min) / (max - min)).collect(); // Normaliza os dados
    (normalized, min, max) // Retorna os dados normalizados, o valor mínimo e o valor máximo
}

fn main() {
    // Dados JSON fornecidos
    let json_data = r#"[{"code":"EUR","codein":"USD","name":"Euro/Dólar Americano","high":"1.0411","low":"1.0305","varBid":"-0.0053","pctChange":"-0.51","bid":"1.0329","ask":"1.0331","timestamp":"1738956489","create_date":"2025-02-07 16:28:09"},{"high":"1.0387","low":"1.0382","varBid":"0.0002","pctChange":"0.02","bid":"1.0384","ask":"1.0386","timestamp":"1738886388"}, ...]"#;

    // Parse dos dados JSON
    let parsed_data: Vec<Value> = serde_json::from_str(json_data).expect("Invalid JSON");

    // Extrair os valores de 'bid'
    let bid_prices: Vec<f64> = parsed_data
        .iter()
        .map(|entry| entry["bid"].as_str().unwrap().parse::<f64>().unwrap())
        .collect();

    // Normalizar os dados
    let (normalized_prices, min_price, max_price) = normalize(&bid_prices);

    // Criar dados de treinamento
    let window_size = 5; // Usar os últimos 5 dias como entrada
    let mut training_data = Vec::new();
    for i in window_size..normalized_prices.len() {
        let inputs: Vec<f64> = normalized_prices[i - window_size..i].to_vec();
        let target = normalized_prices[i];
        training_data.push((inputs, target));
    }

    // Criar e treinar a MLP
    let layer_sizes = &[window_size, 20, 10, 1]; // Arquitetura mais profunda
    let mut mlp = MLP::new(layer_sizes);
    mlp.train(&training_data, 0.005, 20000); // Taxa de aprendizado ajustada e mais épocas

    // Prever o próximo valor (dia 30)
    let last_inputs: Vec<f64> = normalized_prices[normalized_prices.len() - window_size..].to_vec();
    let normalized_prediction = mlp.forward(&last_inputs).last().unwrap()[0];

    // Desnormalizar a previsão
    let denormalized_prediction = normalized_prediction * (max_price - min_price) + min_price;

    println!(
        "Previsão para o bid price em 08/02/2025: {:.4}",
        denormalized_prediction
    );
}


// cd gpt
// cargo run
// cargo run --bin gpt
// cargo.exe "run", "--package", "gpt", "--bin", "gpt"