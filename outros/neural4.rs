use std::collections::HashMap;

struct TextNeuralNetwork {
    weights: HashMap<(usize, usize), f64>,
}

impl TextNeuralNetwork {
    fn new() -> Self {
        TextNeuralNetwork {
            weights: HashMap::new(),
        }
    }

    fn train(&mut self, inputs: &[&str], outputs: &[&str]) {
        for (i, input) in inputs.iter().enumerate() {
            let _output = outputs[i]; // Marking as unused with an underscore
            let delta = 1.0; // You might need to define a more sophisticated update rule.
            for (j, char_input) in input.chars().enumerate() {
                let entry = self.weights.entry((j, char_input as usize)).or_insert(0.0);
                *entry += delta;
            }
        }
    }

    fn generate_response(&self, question: &str) -> Option<String> {
        let mut response = String::new();
        for char_input in question.chars() {
            if let Some(weight) = self.weights.get(&(0, char_input as usize)) {
                response.push(char_input);
                response.push_str(&format!(" (weight: {:.2})", weight));
            }
        }
        if response.is_empty() {
            None
        } else {
            Some(response)
        }
    }
}

fn main() {
    let mut text_network = TextNeuralNetwork::new();

    // Define training data
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

    // Train the text neural network
    for i in 0..inputs.len() {
        text_network.train(&[inputs[i]], &[outputs[i]]);
    }

    // Test the text neural network
    let question = "What is your name?";
    if let Some(response) = text_network.generate_response(question) {
        println!("Question: {}", question);
        println!("Response: {}", response);
    } else {
        println!("Question not recognized.");
    }
}