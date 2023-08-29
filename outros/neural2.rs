use std::collections::HashMap;

struct TextNeuralNetwork {
    weights: HashMap<(usize, char), f64>,
}

impl TextNeuralNetwork {
    fn new() -> Self {
        TextNeuralNetwork {
            weights: HashMap::new(),
        }
    }
	
    fn train(&mut self, inputs: &[&str], outputs: &[&str]) {
        for (i, input) in inputs.iter().enumerate() {
            let output = outputs[i];
            let delta = 1.0; // Voce pode precisar definir uma regra de atualizacao mais sofisticada.
            for (j, char_input) in input.chars().enumerate() {
                let entry = self.weights.entry((j, char_input)).or_insert(0.0);
                *entry += delta;
            }
        }
    }

    fn generate_response(&self, question: &str) -> Option<String> {
        let mut response = String::new();
        let mut total_weight = 0.0;
        for (i, char_input) in question.chars().enumerate() {
            if let Some(weight) = self.weights.get(&(i, char_input)) {
                response.push(char_input);
                total_weight += weight;
            }
        }
        if response.is_empty() {
            None
        } else {
            let word_weights: f64 = response.chars().map(|char| {
                self.weights.get(&(0, char)).copied().unwrap_or(0.0)
            }).sum();
            Some(format!("{} (weight: {:.2})", response, word_weights))
        }
    }
}

fn main() {
    let mut text_network = TextNeuralNetwork::new();

    // Defina os dados de treinamento
    let inputs = vec![
        "What is your name?",
        "How does this work?",
        "Who created you?",
        "What can you do?",
        "Tell me a joke.",
    ];
    let outputs = vec![
        "I am a text neural network.",
        "This code uses simple weights.",
        "I was created by OpenAI.",
        "I can assist with various tasks.",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
    ];

    // Treine a rede neural de texto
    for i in 0..inputs.len() {
        text_network.train(&[inputs[i]], &[outputs[i]]);
    }

    // Teste a rede neural de texto
    let question = "What is your name?";
    if let Some(response) = text_network.generate_response(question) {
        println!("Pergunta: {}", question);
        println!("Resposta: {}", response);
    } else {
        println!("Pergunta nao reconhecida.");
    }
}