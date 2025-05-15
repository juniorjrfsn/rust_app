use ndarray::{Array1, Array2, Array3, s};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use std::fs::{self, File};
use std::io::{self, Write, Read};
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use ab_glyph::{FontRef, PxScale as AbScale};

#[derive(Serialize, Deserialize)]
struct TextClassifier {
    cnn: CNN,
    char_classes: Vec<char>,
}

#[derive(Serialize, Deserialize)]
struct CNN {
    conv_filters: usize,
    filter_size: usize,
    output_size: usize,
    conv_weights: Array3<f64>,
    conv_bias: Array1<f64>,
    fc_weights: Array2<f64>,
    fc_bias: Array1<f64>,
    learning_rate: f64,
    momentum: f64,
    conv_weight_vel: Array3<f64>,
    conv_bias_vel: Array1<f64>,
    fc_weight_vel: Array2<f64>,
    fc_bias_vel: Array1<f64>,
}

impl CNN {
    fn new(conv_filters: usize, filter_size: usize, output_size: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, (2.0 / (filter_size * filter_size) as f64).sqrt()).unwrap();

        let conv_weights = Array3::from_shape_fn(
            (conv_filters, filter_size, filter_size),
            |_| normal.sample(&mut rng),
        );
        let conv_bias = Array1::zeros(conv_filters);
        let fc_input_size = conv_filters * 12 * 12;
        let fc_normal = Normal::new(0.0, (2.0 / fc_input_size as f64).sqrt()).unwrap();
        let fc_weights = Array2::from_shape_fn((output_size, fc_input_size), |_| {
            fc_normal.sample(&mut rng)
        });
        let fc_bias = Array1::zeros(output_size);

        CNN {
            conv_filters,
            filter_size,
            output_size,
            conv_weights,
            conv_bias,
            fc_weights,
            fc_bias,
            learning_rate: 0.005,  // Adjusted learning rate
            momentum: 0.9,
            conv_weight_vel: Array3::zeros((conv_filters, filter_size, filter_size)),
            conv_bias_vel: Array1::zeros(conv_filters),
            fc_weight_vel: Array2::zeros((output_size, fc_input_size)),
            fc_bias_vel: Array1::zeros(output_size),
        }
    }

    fn forward(&self, image: &Array2<f64>) -> (Array1<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>, Array1<f64>) {
        let mut conv_outputs = Vec::new();
        for k in 0..self.conv_filters {
            let filter = self.conv_weights.slice(s![k, .., ..]).to_owned();
            let bias = self.conv_bias[k];
            conv_outputs.push(self.convolve(image, &filter, bias));
        }
        let pooled_outputs: Vec<_> = conv_outputs.iter().map(|conv| self.maxpool(conv)).collect();
        let flattened = Array1::from_iter(pooled_outputs.iter().flat_map(|p| p.iter().cloned()));
        let mut output = self.fc_weights.dot(&flattened) + &self.fc_bias;
        self.softmax(&mut output);
        (output, conv_outputs, pooled_outputs, flattened)
    }

    fn convolve(&self, image: &Array2<f64>, filter: &Array2<f64>, bias: f64) -> Array2<f64> {
        let (h, w) = (image.shape()[0], image.shape()[1]);
        let (fh, fw) = (filter.shape()[0], filter.shape()[1]);
        let (out_h, out_w) = (h - fh + 1, w - fw + 1);
        let mut output = Array2::zeros((out_h, out_w));
        for i in 0..out_h {
            for j in 0..out_w {
                let region = image.slice(s![i..i + fh, j..j + fw]);
                let sum: f64 = region.iter().zip(filter.iter()).map(|(&r, &f)| r * f).sum();
                output[[i, j]] = (sum + bias).max(0.0);
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
                let region = input.slice(s![i * 2..i * 2 + 2, j * 2..j * 2 + 2]);
                output[[i, j]] = region.iter().fold(f64::MIN, |a, &b| a.max(b));
            }
        }
        output
    }

    fn softmax(&self, x: &mut Array1<f64>) {
        let max = x.iter().fold(f64::MIN, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max).exp());
        let sum = exp_x.sum();
        *x = exp_x / sum;
    }

    fn train(&mut self, image: &Array2<f64>, target: &Array1<f64>) {
        let (output, conv_outputs, pooled_outputs, flattened) = self.forward(image);
        let output_error = output - target;

        // Fully connected layer gradients
        let fc_weight_grad = output_error.clone()
            .to_shape((self.output_size, 1))
            .unwrap()
            .dot(&flattened.to_shape((1, flattened.len())).unwrap());
        let fc_bias_grad = output_error.clone();

        // Backpropagation
        let fc_error = self.fc_weights.t().dot(&output_error);
        let pool_size = pooled_outputs[0].len();
        let fc_error_reshaped = fc_error.to_shape((self.conv_filters, pool_size)).unwrap();

        let mut conv_weight_grads = Vec::new();
        let mut conv_bias_grads = Vec::new();

        for k in 0..self.conv_filters {
            let pooled = &pooled_outputs[k];
            let conv = &conv_outputs[k];
            let mut conv_error = Array2::zeros(conv.raw_dim());

            for i in 0..pooled.shape()[0] {
                for j in 0..pooled.shape()[1] {
                    let region = conv.slice(s![i * 2..i * 2 + 2, j * 2..j * 2 + 2]);
                    let max_idx = region.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();
                    conv_error[[i * 2 + max_idx / 2, j * 2 + max_idx % 2]] = 
                        fc_error_reshaped[[k, i * pooled.shape()[1] + j]];
                }
            }

            let mut conv_grad = Array2::zeros((self.filter_size, self.filter_size));
            for i in 0..self.filter_size {
                for j in 0..self.filter_size {
                    let region = image.slice(s![i..i + self.filter_size, j..j + self.filter_size]);
                    conv_grad[[i, j]] = region.iter()
                        .zip(conv_error.iter())
                        .map(|(&r, &e)| r * e)
                        .sum::<f64>();
                }
            }
            conv_weight_grads.push(conv_grad);
            conv_bias_grads.push(conv_error.sum());
        }

        // Update parameters
        for k in 0..self.conv_filters {
            let conv_grad = &conv_weight_grads[k];
            let bias_grad = conv_bias_grads[k];

            let weight_vel = self.conv_weight_vel.slice(s![k, .., ..]).to_owned();
            let weight_vel_update = weight_vel * self.momentum - (conv_grad * self.learning_rate);
            self.conv_weight_vel.slice_mut(s![k, .., ..]).assign(&weight_vel_update);

            let weights = self.conv_weights.slice(s![k, .., ..]).to_owned();
            let updated_weights = weights + &weight_vel_update;
            self.conv_weights.slice_mut(s![k, .., ..]).assign(&updated_weights);

            self.conv_bias_vel[k] = self.momentum * self.conv_bias_vel[k] - self.learning_rate * bias_grad;
            self.conv_bias[k] += self.conv_bias_vel[k];
        }

        self.fc_weight_vel = &self.fc_weight_vel * self.momentum - &(fc_weight_grad * self.learning_rate);
        self.fc_weights += &self.fc_weight_vel;
        self.fc_bias_vel = &self.fc_bias_vel * self.momentum - &(fc_bias_grad * self.learning_rate);
        self.fc_bias += &self.fc_bias_vel;
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
        let (output, _, _, _) = self.cnn.forward(image);
        let (idx, &confidence) = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (self.char_classes[idx], confidence)
    }
}

fn generate_char_image(ch: char, size: u32, font_path: &str) -> Result<RgbImage, io::Error> {
    let mut img = ImageBuffer::from_fn(size, size, |_, _| Rgb([255, 255, 255]));
    let font_data = fs::read(font_path)?;
    let font = FontRef::try_from_slice(&font_data)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let scale = AbScale {
        x: size as f32 * 0.7,
        y: size as f32 * 0.7,
    };
    draw_text_mut(
        &mut img,
        Rgb([0, 0, 0]),
        (size / 4) as i32,
        (size / 4) as i32,
        scale,
        &font,
        &ch.to_string(),
    );
    Ok(img)
}

fn normalize_image(img: &RgbImage) -> Array2<f64> {
    Array2::from_shape_fn(
        (img.height() as usize, img.width() as usize),
        |(y, x)| 1.0 - (img.get_pixel(x as u32, y as u32)[0] as f64 / 255.0),
    )
}

fn list_fonts(font_dir: &str) -> Vec<String> {
    fs::read_dir(font_dir)
        .unwrap_or_else(|e| panic!("Failed to read font directory {}: {}", font_dir, e))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map_or(false, |ext| ext == "ttf") {
                Some(path.to_str()?.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn train_model() -> io::Result<()> {
    println!("Starting training...");
    let char_classes: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:,;'\"(!?)+-*/=".chars().collect();
    let font_dir = "dados/FontsTrain";
    let fonts = list_fonts(font_dir);

    if fonts.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "No font files found in dados/FontsTrain"));
    }

    println!("Found {} training fonts", fonts.len());
    let mut classifier = TextClassifier {
        cnn: CNN::new(16, 5, char_classes.len()),
        char_classes: char_classes.clone(),
    };

    for epoch in 0..100 {  // Increased epochs
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for font_path in &fonts {
            for (i, &ch) in char_classes.iter().enumerate() {
                let img = generate_char_image(ch, 28, font_path)?;
                let input = normalize_image(&img);
                let mut target = Array1::zeros(char_classes.len());
                target[i] = 1.0;

                let (output, _, _, _) = classifier.cnn.forward(&input);
                classifier.cnn.train(&input, &target);

                total_loss += -target.iter()
                    .zip(output.iter())
                    .map(|(&t, &o)| t * o.max(1e-10).ln())
                    .sum::<f64>();

                let predicted = output.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                if predicted == i {
                    correct += 1;
                }
                total += 1;
            }
        }
        let accuracy = correct as f64 / total as f64;
        println!("Epoch {}: Loss {:.4}, Accuracy {:.2}%", epoch, total_loss / total as f64, accuracy * 100.0);
    }

    std::fs::create_dir_all("dados")?;
    classifier.save("dados/text_classifier.json")?;
    println!("Training completed successfully");
    Ok(())
}

fn test_model() -> io::Result<()> {
    let classifier = TextClassifier::load("dados/text_classifier.json")
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to load model: {}", e)))?;
    let test_chars = ['A', 'B', 'C', 'J', 'K', 'L', 'X', 'Y', 'Z'];
    let font_dir = "dados/FontsTest";
    let fonts = list_fonts(font_dir);

    if fonts.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "No font files found in dados/FontsTest"));
    }

    println!("Found {} test fonts", fonts.len());
    let mut correct = 0;
    let mut total = 0;

    for ch in test_chars.iter() {
        for font_path in &fonts {
            let img = generate_char_image(*ch, 28, font_path)?;
            let input = normalize_image(&img);
            let (predicted, confidence) = classifier.predict(&input);

            println!(
                "Char: {} | Font: {} | Predicted: {} | Confidence: {:.2}%",
                ch,
                font_path,
                predicted,
                confidence * 100.0
            );

            if *ch == predicted {
                correct += 1;
            }
            total += 1;
        }
    }
    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("Test Accuracy: {:.2}% ({} out of {})", accuracy, correct, total);
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

    cargo run -- treino     # Trains and saves the model
    cargo run -- reconhecer # Loads and tests the model
*/
