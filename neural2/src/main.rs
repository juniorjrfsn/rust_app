use std::collections::{HashMap, HashSet};
use rand::prelude::*;

extern crate colored;
use colored::*;

struct LanguageModel {
    transitions: HashMap<String, HashMap<String, f64>>,
    start_words: HashSet<String>,
    end_words: HashSet<String>,
}

impl LanguageModel {
    fn new() -> Self {
        LanguageModel {
            transitions: HashMap::new(),
            start_words: HashSet::new(),
            end_words: HashSet::new(),
        }
    }

    fn train(&mut self, training_data: &[&str]) {
        for sentence in training_data {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            // Adicionar palavras de início e fim
            let mut words_with_markers = words.clone();
            words_with_markers.insert(0, "<START>");
            words_with_markers.push("<END>");

            for (i, &word) in words_with_markers.iter().enumerate() {
                let next_word = words_with_markers.get(i + 1).unwrap_or(&"<END>");

                // Registrar palavra de início
                if i == 0 {
                    self.start_words.insert(word.to_string());
                }

                // Registrar palavra de fim
                if i == words_with_markers.len() - 1 {
                    self.end_words.insert(next_word.to_string());
                }

                // Atualizar transições
                let entry = self
                    .transitions
                    .entry(word.to_string())
                    .or_insert(HashMap::new());
                let count = entry.entry(next_word.to_string()).or_insert(0.0);
                *count += 1.0;
            }
        }

        // Normalizar probabilidades
        for (_, word_transitions) in self.transitions.iter_mut() {
            let total_count: f64 = word_transitions.values().sum();
            for (_, count) in word_transitions.iter_mut() {
                *count /= total_count;
            }
        }
    }

    fn generate_response(&self, seed_word: &str, max_length: usize) -> String {
        let mut rng = thread_rng();
        let mut response = String::new();
        let mut current_word = seed_word.to_string();

        for _ in 0..max_length {
            // Obter probabilidades de transição
            let next_word_probabilities = self.transitions.get(&current_word).unwrap();

            // Selecionar próxima palavra com base em probabilidades
            let mut total_probability = 0.0;
            let mut next_word = "";
            let random_value = rng.gen_range(0.0..1.0);
            for (word, probability) in next_word_probabilities.iter() {
                total_probability += probability;
                if random_value <= total_probability {
                    next_word = word;
                    break;
                }
            }

            // Sair do loop se palavra de fim for selecionada
            if self.end_words.contains(next_word) {
                break;
            }

            // Adicionar próxima palavra à resposta
            response.push_str(" ");
            response.push_str(next_word);

            current_word = next_word.to_string();
        }

        response
    }
}

fn main() {
    let mut language_model = LanguageModel::new();

    // Carregar dados de treinamento (substitua por seus dados)
    let training_data = vec![
        "Take the hassle out of icons in your website.
        Font Awesome is the Internet's icon library and toolkit, used by millions of designers, developers, and content creators.",
        "LATEST UPDATES
        Discover what's new in Font Awesome.
        NEW Introducing Font Awesome Sharp Light!
        Font Awesome Sharp Light has arrived! It's perfect for when you need to add a breezy, light touch of modern elegance. Read the announcement and see what's new in the 6.4.2 release.

        BETA Explore the Icon Wizard!
        With our new Icon Wizard, you can magically add a modifier  like circle-plus, slash, or even poop  to almost any Font Awesome icon. Available now to Font Awesome Pro subscribers.

        More Plugins + Packages
        Font Awesome 6 makes it even easier to use icons where you want to. More plugins and packages to match your stack. Less time wrestling browser rendering.

        Version 6.4.2
        26,233 Pro Icons
        68 Categories
        2,025 Free Icons",
        "Build fast, responsive sites with Bootstrap
        Powerful, extensible, and feature-packed frontend toolkit. Build and customize with Sass, utilize prebuilt grid system and components, and bring projects to life with powerful JavaScript plugins.

        npm i bootstrap@5.3.1
        Currently v5.3.1  Download  All releases

        Get started any way you want
        Jump right into building with Bootstrap?use the CDN, install it via package manager, or download the source code.

        Read installation docs
        Install via package manager
        Install Bootstrap s source Sass and JavaScript files via npm, RubyGems, Composer, or Meteor. Package managed installs dont include documentation or our full build scripts. You can also use any demo from our Examples repo to quickly jumpstart Bootstrap projects.

        npm install bootstrap@5.3.1

        gem install bootstrap -v 5.3.1
        Read our installation docs for more info and additional package managers.

        Include via CDN
        When you only need to include Bootstraps compiled CSS or JS, you can use jsDelivr. See it in action with our simple quick start, or browse the examples to jumpstart your next project. You can also choose to include Popper and our JS separately.

       Read our getting started guides
        Get a jump on including Bootstrap s source files in a new project with our official guides.
        Webpack
        Parcel
        Vite
        Customize everything with Sass
        Bootstrap utilizes Sass for a modular and customizable architecture. Import only the components you need, enable global options like gradients and shadows, and write your own CSS with our variables, maps, functions, and mixins.",
        "Build and extend in real-time with CSS variables
        Bootstrap 5 is evolving with each release to better utilize CSS variables for global theme styles, individual components, and even utilities. We provide dozens of variables for colors, font styles, and more at a :root level for use anywhere. On components and utilities, CSS variables are scoped to the relevant class and can easily be modified.",

        // ... Adicione mais frases para treinamento ...
    ];

    // Treinar o modelo
    language_model.train(&training_data);

    // Gerar resposta
    let seed_word = "Pergunta:";
    let seed_ = "dfsd";
    println!("{} {}", seed_word.blue().bold(),seed_.magenta().bold());
    let response = language_model.generate_response(seed_, 20);
    let response_ = "Resposta:";
    println!("{} {}",response_.magenta().bold(), response.red().bold());
}