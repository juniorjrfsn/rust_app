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

            let possible_next_words = self
                .transitions
                .get(&current_word)
                .unwrap()
                .iter()
                .collect::<Vec<&String>>();

            current_word = possible_next_words[rng.gen_range(0..possible_next_words.len())].to_string();
        }

        response
    }
}

fn main() {
    let mut language_model = LanguageModel::new();

    // Define training data in Portuguese
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

    // Train the language model
    language_model.train(&training_data);

    // Test the language model
    let seed_word = "Awesome";
    let response = language_model.generate_response(seed_word, 30);
    println!("Seed: {}", seed_word);
    println!("Response: {}", response);
}
