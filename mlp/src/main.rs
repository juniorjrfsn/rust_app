use image::{open, GenericImage, GenericImageView, imageops::resize_exact};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::path::Path;
use tch::{nn, Tensor};
use tch::vision::cross_entropy_for_logits;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Imagem {
    dados: Vec<f32>,
    label: usize,
}

fn carregar_imagem(caminho: &str) -> Result<Imagem, Box<dyn std::error::Error>> {
    let img = image::open(caminho)?.to_rgb8();
    let (largura, altura) = img.dimensions();
    let dados: Vec<f32> = img.pixels()
        .flat_map(|p| vec![p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
        .collect();
    Ok(Imagem { dados, label: 0 }) // Rótulo precisa ser definido externamente
}

fn carregar_dados(diretorio: &str) -> Result<Vec<Imagem>, Box<dyn std::error::Error>> {
    let mut imagens: Vec<Imagem> = Vec::new();
    for entrada in std::fs::read_dir(diretorio)? {
        let entrada = entrada?;
        let caminho_imagem = entrada.path();
        if let Some(ext) = caminho_imagem.extension() {
            if ext == "jpg" || ext == "jpeg" || ext == "png" {
                match carregar_imagem(caminho_imagem.to_str().unwrap()) {
                    Ok(mut imagem) => {
                        // Extrair o rótulo do nome do diretório (exemplo: "img_train/classe_1/imagem.jpg" -> rótulo 1)
                        if let Some(nome_diretorio) = caminho_imagem.parent().and_then(|p| p.file_name()) {
                            if let Some(nome_diretorio_str) = nome_diretorio.to_str() {
                                if let Ok(label) = nome_diretorio_str.split('_').last().unwrap_or("0").parse::<usize>() {
                                    imagem.label = label;
                                }
                            }
                        }
                        imagens.push(imagem);
                    }
                    Err(e) => eprintln!("Erro ao carregar {}: {}", caminho_imagem.display(), e),
                }
            }
        }
    }
    imagens.shuffle(&mut rand::thread_rng());
    Ok(imagens)
}

fn criar_modelo(vs: &nn::VarStore) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 3, 16, 3, Default::default()))
        .add(nn::relu())
        .add(nn::max_pool2d(2, Default::default()))
        .add(nn::conv2d(vs, 16, 32, 3, Default::default()))
        .add(nn::relu())
        .add(nn::max_pool2d(2, Default::default()))
        .add(nn::flatten())
        .add(nn::linear(vs, 32 * 7 * 7, 10)) // Ajuste conforme necessário
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let imagens = carregar_dados("img_train")?;

    let n_treino = (imagens.len() as f64 * 0.8) as usize;
    let (treino, validacao) = imagens.split_at(n_treino);

    // Configuração para usar a CPU
    let vs = nn::VarStore::new(tch::Device::Cpu); 
    let model = criar_modelo(&vs);

    let opt = tch::optim::Adam::new(&vs, 1e-3)?;

    for epoch in 1..=10 {
        let mut perda_total = 0.0;
        for imagem in treino {
            let dados_tensor = Tensor::from_slice(&imagem.dados).view((1, 3, 28, 28)); // Redimensionar para 28x28
            let label_tensor = Tensor::from_slice(&[imagem.label as i64]);

            let output = model.forward(&dados_tensor);
            let perda = cross_entropy_for_logits(&output, &label_tensor);

            opt.backward_step(&perda);
            perda_total += perda.double_value(&[]);
        }

        println!("Época: {}, Perda: {}", epoch, perda_total);

        // Validação (opcional)
        // ...
    }

    vs.save("modelo_treinado.tch")?;

    Ok(())
}