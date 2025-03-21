use ndarray::{Array1, Array2, Array3, s};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, Write, Read};
use image::{ImageBuffer, Rgb, RgbImage};

#[derive(Serialize, Deserialize)]
struct TextClassifier {
    cnn: CNN,
    char_classes: Vec<char>,
}

#[derive(Serialize, Deserialize)]
struct CNN {
    conv_filters: usize,
    filter_size: usize,
    hidden_size: usize,
    output_size: usize,
    conv_weights: Array3<f64>,
    conv_bias: Array1<f64>,
    fc_weights: Array2<f64>,
    fc_bias: Array1<f64>,
    learning_rate: f64,
}

impl CNN {
    fn new(conv_filters: usize, filter_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let conv_weights = Array3::from_shape_fn(
            (conv_filters, filter_size, filter_size),
            |_| normal.sample(&mut rng)
        );
        let conv_bias = Array1::zeros(conv_filters);
        
        let fc_input_size = conv_filters * 12 * 12;
        let fc_weights = Array2::from_shape_fn(
            (output_size, fc_input_size),
            |_| normal.sample(&mut rng)
        );
        let fc_bias = Array1::zeros(output_size);

        CNN {
            conv_filters,
            filter_size,
            hidden_size,
            output_size,
            conv_weights,
            conv_bias,
            fc_weights,
            fc_bias,
            learning_rate: 0.001,
        }
    }

    fn forward(&self, image: &Array2<f64>) -> Array1<f64> {
        let mut conv_outputs = Vec::new();
        for k in 0..self.conv_filters {
            let filter = self.conv_weights.slice(s![k, .., ..]).to_owned();
            let bias = self.conv_bias[k];
            conv_outputs.push(self.convolve(image, &filter, bias));
        }

        let pooled_outputs: Vec<_> = conv_outputs.iter().map(|conv| self.maxpool(conv)).collect();
        
        let flattened = Array1::from_iter(
            pooled_outputs.iter().flat_map(|p| p.iter().cloned())
        );

        let mut output = self.fc_weights.dot(&flattened) + &self.fc_bias;
        self.softmax(&mut output);
        output
    }

    fn convolve(&self, image: &Array2<f64>, filter: &Array2<f64>, bias: f64) -> Array2<f64> {
        let (h, w) = (image.shape()[0], image.shape()[1]);
        let (fh, fw) = (filter.shape()[0], filter.shape()[1]);
        let (out_h, out_w) = (h - fh + 1, w - fw + 1);
        
        let mut output = Array2::zeros((out_h, out_w));
        for i in 0..out_h {
            for j in 0..out_w {
                let region = image.slice(s![i..i+fh, j..j+fw]);
                output[[i, j]] = region.iter()
                    .zip(filter.iter())
                    .map(|(&r, &f)| r * f)
                    .sum::<f64>()
                    .max(0.0) + bias;
            }
        }
        output
    }

    fn maxpool(&self, input: &Array2<f64>) -> Array2<f64> {
        let (h, w) = (input.shape()[0], input.shape()[1]);
        let (out_h, out_w) = (h / 2, w / 2);
        let mut output = Array2::zeros((out_h, out_w));
        
        for i in 0..out_h {
            for j in 0..out_w {
                let region = input.slice(s![i*2..i*2+2, j*2..j*2+2]);
                output[[i, j]] = region.iter().fold(f64::MIN, |a, &b| a.max(b));
            }
        }
        output
    }

    fn softmax(&self, x: &mut Array1<f64>) {
        let max = x.iter().fold(f64::MIN, |a, &b| a.max(b));
        *x = x.mapv(|v| (v - max).exp());
        let sum = x.sum();
        *x = x.mapv(|v| v / sum);
    }

    fn train(&mut self, image: &Array2<f64>, target: &Array1<f64>) {
        let output = self.forward(image);
        let error = &output - target;

        let flattened_size = self.fc_weights.shape()[1];
        let flattened = Array2::from_shape_vec(
            (1, flattened_size),
            vec![1.0; flattened_size]
        ).unwrap();
        
        let fc_weight_grad = error.clone().to_shape((self.output_size, 1)).unwrap().dot(&flattened);
        self.fc_weights -= &(fc_weight_grad * self.learning_rate);
        self.fc_bias -= &(error * self.learning_rate);
    }
}

impl TextClassifier {
    fn save(&self, path: &str) -> io::Result<()> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(serde_json::from_str(&contents)?)
    }

    fn predict(&self, image: &Array2<f64>) -> (char, f64) {
        let output = self.cnn.forward(image);
        let (idx, &confidence) = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (self.char_classes[idx], confidence)
    }
}

fn generate_char_image(_ch: char, size: u32) -> RgbImage {
    let mut img = ImageBuffer::new(size, size);
    for pixel in img.pixels_mut() {
        *pixel = if rand::random::<f64>() < 0.5 { 
            Rgb([255, 255, 255]) 
        } else { 
            Rgb([0, 0, 0]) 
        };
    }
    img
}

fn normalize_image(img: &RgbImage) -> Array2<f64> {
    Array2::from_shape_fn(
        (img.height() as usize, img.width() as usize),
        |(y, x)| img.get_pixel(x as u32, y as u32)[0] as f64 / 255.0
    )
}

fn train_model() -> io::Result<()> {
    println!("Starting training...");
    let char_classes: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    let mut classifier = TextClassifier {
        cnn: CNN::new(8, 5, 128, char_classes.len()),
        char_classes: char_classes.clone(),
    };

    for epoch in 0..10 {
        let mut total_loss = 0.0;
        for (i, &ch) in char_classes.iter().enumerate() {
            let img = generate_char_image(ch, 28);
            let input = normalize_image(&img);
            let mut target = Array1::zeros(char_classes.len());
            target[i] = 1.0;
            
            classifier.cnn.train(&input, &target);
            let output = classifier.cnn.forward(&input);
            total_loss += -output[i].ln();
        }
        println!("Epoch {}: Loss {}", epoch, total_loss);
    }

    std::fs::create_dir_all("dados").map_err(|e| {
        println!("Failed to create directory: {}", e);
        e
    })?;
    classifier.save("dados/text_classifier.json").map_err(|e| {
        println!("Failed to save model: {}", e);
        e
    })?;
    println!("Training completed successfully");
    Ok(())
}

fn test_model() -> io::Result<()> {
    let classifier = TextClassifier::load("dados/text_classifier.json")?;
    let test_chars = ['A', 'B', 'C'];
    
    for ch in test_chars.iter() {
        let img = generate_char_image(*ch, 28);
        let input = normalize_image(&img);
        let (predicted, confidence) = classifier.predict(&input);
        println!("Char: {} | Predicted: {} | Confidence: {:.2}%", 
            ch, predicted, confidence * 100.0);
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "treino" => train_model(),
            "reconhecer" => test_model(),
            _ => {
                println!("Usage: {} [treino|reconhecer]", args[0]);
                Ok(())
            }
        }
    } else {
        println!("Usage: {} [treino|reconhecer]", args[0]);
        Ok(())
    }
}

/*
cd cnn_reconhece_texto

cargo run -- treino    # Trains and saves the model
cargo run -- reconhecer # Loads and tests the model
*/
