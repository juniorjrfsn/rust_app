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

        for _ in 0..max_length {
            response.push_str(&current_word);
            response.push(' ');

            if let Some(possible_next_words) = self.transitions.get(&current_word) {
                let next_words: Vec<&String> = possible_next_words.iter().collect();
                if !next_words.is_empty() {
                    let random_index = rng.gen_range(0..next_words.len());
                    current_word = next_words[random_index].to_string();
                } else {
                    break; // No more possible words
                }
            } else {
                break; // Current word not found in transitions
            }
        }

        response
    }
}

fn main() {
    let mut language_model = LanguageModel::new();

    // Define training data
    let training_data = vec![
        "Qual eh o seu nome?",
        "Como voce trabalha?",
        "What are you doing?",
        "Can you help me?",
        "Meu nome eh junior",
        "Sim eu posso",
    ];

    // Train the language model
    language_model.train(&training_data);

    // Test the language model
    let seed_word = "Qual";
    let response = language_model.generate_response(seed_word, 10);
    println!("Seed: {}", seed_word);
    println!("Response: {}", response);
}
