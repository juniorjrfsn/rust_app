mod language_model {
    use std::collections::HashMap;

    pub struct Model {
        data: HashMap<String, String>,
        pub max_prediction: String,
    }

    impl Model {
        pub fn new() -> Self {
            Model {
                data: HashMap::new(),
                max_prediction: String::new(),
            }
        }

        pub fn train(&mut self, training_data: &HashMap<String, String>) {
            self.data = training_data.clone();
        }
        pub fn predict(&self, _input: String) -> String {
            return  match self.data.get(_input.as_str()) {
                Some(prediction) => prediction.to_string(),
                None => format!("A palavra '{}' nao esta na lista de treinamento", _input),
            };
        }
    }
}

fn main() {
    use std::collections::HashMap;
    let mut training_data: HashMap<String, String>= HashMap::new();

    training_data.insert(String::from("Ola") , String::from("mundo") );
    training_data.insert(String::from("Como"), String::from("vai") );
    training_data.insert(String::from("Voce"), String::from("esta") );
    training_data.insert(String::from("Bem?"), String::from("sim") );

    let mut model = language_model::Model::new();
    model.train(&training_data );
    let input = String::from("Ola");
    let prediction = model.predict(input.clone());


    // let prediction = decoder.predict(encoder.forward(input.as_bytes()));

    println!("A palavra '{}' foi prevista para a palavra: {}",prediction, input);
    println!("Gerando uma frase : '{} {}' ",input, prediction);
}