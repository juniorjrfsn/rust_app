use rand::Rng;

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
}

fn main() {
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

    let test_all_zeros = vec![0.0; INPUT_SIZE];
    let test_all_ones = vec![1.0; INPUT_SIZE];
    let test_half_gray = vec![0.5; INPUT_SIZE];
    let test_quarter_gray = vec![0.25; INPUT_SIZE];

    println!("Prediction for all zeros: {}", perceptron.predict(&test_all_zeros));
    println!("Prediction for all ones: {}", perceptron.predict(&test_all_ones));
    println!("Prediction for half gray: {}", perceptron.predict(&test_half_gray));
    println!("Prediction for quarter gray: {}", perceptron.predict(&test_quarter_gray));

    println!("Final perceptron state: {:?}", perceptron);
}