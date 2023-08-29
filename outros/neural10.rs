use std::collections::{HashMap, HashSet};
use rand::prelude::*;

struct LanguageModel {
    transitions: HashMap<String, HashSet<String>>,
}

impl LanguageModel {
    fn new() -> Self {
        LanguageModel {
            transitions: HashMap::new(),
        }
    }

    fn train(&mut self, training_data: &[&str]) {
        for sentence in training_data {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            for (i, &word) in words.iter().enumerate() {
                let next_word = if i < words.len() - 1 {
                    words[i + 1]
                } else {
                    // End of sentence
                    ""
                };
                let entry = self.transitions.entry(word.to_string()).or_insert(HashSet::new());
                entry.insert(next_word.to_string());
            }
        }
    }

    fn generate_response(&self, seed_word: &str, max_length: usize) -> String {
        let mut rng = thread_rng();
        let mut response = String::new();
        let mut current_word = seed_word.to_string();

        while response.len() <= max_length {
            response.push_str(&current_word);
            response.push(' ');

            let possible_next_words = self.transitions.get(&current_word).unwrap();

            current_word = possible_next_words.iter().nth(rng.gen_range(0..possible_next_words.len())).unwrap().to_string();
        }

        response
    }
}

fn main() {
    let mut language_model = LanguageModel::new();

    // Define training data in Portuguese
    let training_data = vec![
        "O ceu estava claro e o ar fresco quando Maria acordou naquela manhan.",
        "Ela se levantou da cama e olhou pela janela, observando os passaros voando ao redor.",
        // ... Adicione mais frases para treinamento ...
    ];

    // Train the language model
    language_model.train(&training_data);

    // Test the language model
    let seed_word = "Ela";
    let response = language_model.generate_response(seed_word, 30);
    println!("Seed: {}", seed_word);
    println!("Response: {}", response);
}
