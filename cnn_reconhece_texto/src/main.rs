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
}

impl CNN {
    fn new(conv_filters: usize, filter_size: usize, output_size: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.1).unwrap();
        let conv_weights = Array3::from_shape_fn(
            (conv_filters, filter_size, filter_size),
            |_| normal.sample(&mut rng),
        );
        let conv_bias = Array1::zeros(conv_filters);
        let fc_input_size = conv_filters * 12 * 12;
        let fc_weights = Array2::from_shape_fn((output_size, fc_input_size), |_| {
            normal.sample(&mut rng)
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
            learning_rate: 0.01,
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
                output[[i, j]] = region.iter().zip(filter.iter()).map(|(&r, &f)| r * f).sum::<f64>().max(0.0) + bias;
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
        *x = x.mapv(|v| (v - max).exp());
        let sum = x.sum();
        *x = x.mapv(|v| v / sum);
    }

    fn train(&mut self, image: &Array2<f64>, target: &Array1<f64>) {
        let (output, conv_outputs, pooled_outputs, flattened) = self.forward(image);
        let output_error = &output - target;

        // Backpropagation through fully connected layer
        let fc_weight_grad = output_error.clone().to_shape((self.output_size, 1)).unwrap()
            .dot(&flattened.to_shape((1, flattened.len())).unwrap());
        let fc_bias_grad = output_error.clone();
        self.fc_weights -= &(fc_weight_grad * self.learning_rate);
        self.fc_bias -= &(fc_bias_grad * self.learning_rate);

        // Backpropagation through pooling and convolution
        let fc_error = self.fc_weights.t().dot(&output_error);
        let mut conv_errors = Vec::new();
        for k in 0..self.conv_filters {
            let pooled = &pooled_outputs[k];
            let conv = &conv_outputs[k];
            let mut conv_error = Array2::zeros(conv.raw_dim());

            // Upsample pooling error
            for i in 0..pooled.shape()[0] {
                for j in 0..pooled.shape()[1] {
                    let region = conv.slice(s![i * 2..i * 2 + 2, j * 2..j * 2 + 2]);
                    let max_idx = region.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();
                    conv_error[[i * 2 + max_idx / 2, j * 2 + max_idx % 2]] = fc_error[k];
                }
            }

            // Update convolution weights
            let mut conv_weight_grad = Array2::zeros((self.filter_size, self.filter_size));
            for i in 0..self.filter_size {
                for j in 0..self.filter_size {
                    let region = image.slice(s![i..i + self.filter_size, j..j + self.filter_size]);
                    conv_weight_grad[[i, j]] = region.iter()
                        .zip(conv_error.iter())
                        .map(|(&r, &e)| r * e)
                        .sum::<f64>();
                }
            }
            let updated_weights = self.conv_weights.slice(s![k, .., ..]).to_owned() - &(conv_weight_grad * self.learning_rate);
            self.conv_weights.slice_mut(s![k, .., ..]).assign(&updated_weights);
            self.conv_bias[k] -= conv_error.sum() * self.learning_rate;
            conv_errors.push(conv_error);
        }
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
        println!("Predicted index: {}, Confidence: {:.4}", idx, confidence);
        (self.char_classes[idx], confidence)
    }
}

fn generate_char_image(ch: char, size: u32, font_path: &str) -> RgbImage {
    let mut img = ImageBuffer::from_fn(size, size, |_, _| Rgb([255, 255, 255]));
    if !fs::metadata(font_path).is_ok() {
        panic!("Font file not found at {}", font_path);
    }
    let font_data = fs::read(font_path).expect("Failed to read font file");
    let font = FontRef::try_from_slice(&font_data).expect("Failed to load font");
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
    img
}

fn normalize_image(img: &RgbImage) -> Array2<f64> {
    Array2::from_shape_fn(
        (img.height() as usize, img.width() as usize),
        |(y, x)| 1.0 - (img.get_pixel(x as u32, y as u32)[0] as f64 / 255.0),
    )
}

fn list_fonts(font_dir: &str) -> Vec<String> {
    let mut fonts = Vec::new();
    for entry in fs::read_dir(font_dir).expect("Failed to read font directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "ttf") {
                fonts.push(path.to_str().unwrap().to_string());
            }
        }
    }
    fonts
}

fn train_model() -> io::Result<()> {
    println!("Starting training...");
    let char_classes: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    let font_dir = "D:\\projetos\\rust_app\\cnn_reconhece_texto\\dados\\FontsTrain";
    let fonts = list_fonts(font_dir);

    let mut classifier = TextClassifier {
        cnn: CNN::new(16, 5, char_classes.len()),
        char_classes: char_classes.clone(),
    };

    for epoch in 0..100 {
        let mut total_loss = 0.0;
        for font_path in &fonts {
            for (i, &ch) in char_classes.iter().enumerate() {
                let img = generate_char_image(ch, 28, font_path);
                let input = normalize_image(&img);
                let mut target = Array1::zeros(char_classes.len());
                target[i] = 1.0;
                classifier.cnn.train(&input, &target);
                let (output, _, _, _) = classifier.cnn.forward(&input);
                total_loss += -output[i].ln();
            }
        }
        println!("Epoch {}: Loss {:.4}", epoch, total_loss);
    }

    std::fs::create_dir_all("dados")?;
    classifier.save("dados/text_classifier.json")?;
    println!("Training completed successfully");
    Ok(())
}

fn test_model() -> io::Result<()> {
    let classifier = TextClassifier::load("dados/text_classifier.json")?;
    let test_chars = ['A', 'B', 'C'];
    let font_dir = "D:\\projetos\\rust_app\\cnn_reconhece_texto\\dados\\FontsTest";
    let fonts = list_fonts(font_dir);

    for ch in test_chars.iter() {
        for font_path in &fonts {
            let img = generate_char_image(*ch, 28, font_path);
            let input = normalize_image(&img);
            let (predicted, confidence) = classifier.predict(&input);
            println!(
                "Char: {} | Font: {} | Predicted: {} | Confidence: {:.2}%",
                ch,
                font_path,
                predicted,
                confidence * 100.0
            );
        }
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
