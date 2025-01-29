use rand::Rng;
use std::io;
use ndarray::prelude::*;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.01 * x // Coeficiente de vazamento
    }
}


fn normalizar_dados(dados: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let dados_array = Array2::from_shape_vec((dados.len(), dados[0].len()), dados.iter().flatten().cloned().collect()).unwrap();
    let mut dados_normalizados = Array2::zeros(dados_array.dim());

    for i in 0..dados_array.ncols() {
        let min_val = dados_array.column(i).iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = dados_array.column(i).iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        dados_normalizados.column_mut(i).iter_mut().zip(dados_array.column(i).iter()).for_each(|(normalizado, &original)| {
            *normalizado = (original - min_val) / (max_val - min_val);
        });
    } 
    // Correctly convert back to Vec<Vec<f32>>
    let mut result = Vec::new();
    for row in dados_normalizados.outer_iter() {
        result.push(row.to_vec());
    }
    result
}

// fn inicializar_pesos(tamanho_entrada: usize, tamanho_saida: usize) -> Vec<Vec<f32>> {
//     let mut rng = rand::rng();
//     (0..tamanho_saida)
//         .map(|_| (0..tamanho_entrada).map(|_| rng.random::<f32>() * (2.0 / tamanho_entrada as f32).sqrt()).collect())
//         .collect()
// }

fn inicializar_pesos(tamanho_entrada: usize, tamanho_saida: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let limite = (6.0 / (tamanho_entrada + tamanho_saida) as f32).sqrt(); // Cálculo do limite
    (0..tamanho_saida)
        .map(|_| (0..tamanho_entrada).map(|_| rng.random_range(-limite..=limite)).collect()) // Pesos entre -limite e +limite
        .collect()
}

fn inicializar_biases(tamanho_saida: usize) -> Vec<f32> {
    vec![0.0; tamanho_saida]
}

fn calcular_saida(entrada: &Vec<f32>, pesos: &(Vec<Vec<f32>>, Vec<Vec<f32>>), biases: &(Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let camada_oculta: Vec<f32> = pesos.0.iter().map(|w| relu(w.iter().zip(entrada).map(|(wi, ei)| wi * ei).sum::<f32>() + biases.0[0])).collect();
    pesos.1.iter().map(|w| sigmoid(w.iter().zip(&camada_oculta).map(|(wi, ci)| wi * ci).sum::<f32>() + biases.1[0])).collect()
}

fn treinar_rede(dados_treinamento: &Vec<(Vec<f32>, Vec<f32>)>, taxa_aprendizagem: f32, epocas: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let (mut pesos_oculta, mut pesos_saida) = (inicializar_pesos(4, 5), inicializar_pesos(5, 3));
    let (mut biases_oculta, mut biases_saida) = (inicializar_biases(5), inicializar_biases(3));

    for _ in 0..epocas {
        for (entrada, saida_desejada) in dados_treinamento {
            // Forward pass
            let camada_oculta: Vec<f32> = pesos_oculta.iter().map(|w| {
                relu(w.iter().zip(entrada).map(|(wi, ei)| wi * ei).sum::<f32>() + biases_oculta[0])
            }).collect();
            let saida_rede: Vec<f32> = pesos_saida.iter().map(|w| {
                sigmoid(w.iter().zip(&camada_oculta).map(|(wi, ci)| wi * ci).sum::<f32>() + biases_saida[0])
            }).collect();

            // Backward pass
            let erro: Vec<f32> = saida_rede.iter().zip(saida_desejada).map(|(s, d)| s - d).collect();
            let gradiente_saida: Vec<f32> = erro.iter().zip(&saida_rede).map(|(e, s)| e * s * (1.0 - s)).collect();

            let mut gradiente_camada_oculta: Vec<f32> = vec![0.0; pesos_oculta.len()]; // Initialize with zeros
            for k in 0..pesos_saida[0].len() { // Iterate through output neurons
                for j in 0..pesos_oculta.len() { // Iterate through hidden neurons
                    for i in 0..pesos_saida.len() { // Iterate through weights connecting hidden to output
                        gradiente_camada_oculta[j] += pesos_saida[i][k] * gradiente_saida[i] * if camada_oculta[j] > 0.0 { 1.0 } else { 0.0 };
                    }
                }
            }

            // Update weights and biases
            for i in 0..pesos_saida.len() {
                for j in 0..pesos_saida[i].len() {
                    pesos_saida[i][j] -= taxa_aprendizagem * gradiente_saida[i] * camada_oculta[j];
                }
                biases_saida[i] -= taxa_aprendizagem * gradiente_saida[i];
            }

            for i in 0..pesos_oculta.len() {
                for j in 0..pesos_oculta[i].len() {
                    pesos_oculta[i][j] -= taxa_aprendizagem * gradiente_camada_oculta[i] * entrada[j];
                }
                biases_oculta[i] -= taxa_aprendizagem * gradiente_camada_oculta[i];
            }
        }
    }
    (pesos_oculta, biases_oculta, pesos_saida, biases_saida)
}

fn obter_entradas() -> Vec<f32> {
    let mut input = String::new();
    println!("Insira as características de saúde:");
    println!("1. Peso (kg)");
    io::stdin().read_line(&mut input).expect("Falha ao ler entrada");
    let peso: f32 = input.trim().parse().expect("Por favor, insira um número válido");
    input.clear();
    println!("2. Altura (m)");
    io::stdin().read_line(&mut input).expect("Falha ao ler entrada");
    let altura: f32 = input.trim().parse().expect("Por favor, insira um número válido");
    input.clear();
    println!("3. Idade (anos)");
    io::stdin().read_line(&mut input).expect("Falha ao ler entrada");
    let idade: f32 = input.trim().parse().expect("Por favor, insira um número válido");
    input.clear();
    println!("4. Nível de atividade física (1-5)");
    io::stdin().read_line(&mut input).expect("Falha ao ler entrada");
    let atividade_fisica: f32 = input.trim().parse().expect("Por favor, insira um número válido");
    normalizar_dados(&vec![vec![peso, altura, idade, atividade_fisica]])[0].to_vec()
}

fn interpretar_saida(saidas: &Vec<f32>) -> (String, String, String) {
    let imc_msg = if saidas[0] < 0.5 {
        "IMC baixo. Pode ser necessário ganhar peso para uma saúde ideal."
    } else if saidas[0] < 0.75 {
        "IMC normal. Mantenha um estilo de vida saudável."
    } else {
        "IMC alto. Pode ser necessário perder peso para uma saúde ideal."
    };

    let recomendacao_msg1 = if saidas[1] < 0.5 {
        "Nível baixo de atividade física recomendado."
    } else if saidas[1] < 0.75 {
        "Nível moderado de atividade física recomendado."
    } else {
        "Nível alto de atividade física recomendado."
    };

    let recomendacao_msg2 = if saidas[2] < 0.5 {
        "Considere atividades físicas leves como caminhada."
    } else if saidas[2] < 0.75 {
        "Considere atividades físicas moderadas como ciclismo."
    } else {
        "Considere atividades físicas intensas como corrida."
    };

    (imc_msg.to_string(), recomendacao_msg1.to_string(), recomendacao_msg2.to_string())
}

fn main() {
    // let dados_treinamento: Vec<(Vec<f32>, Vec<f32>)> = vec![
    //     (vec![70.0, 1.75, 30.0, 3.0], vec![0.6, 0.6, 0.6]),
    //     (vec![90.0, 1.65, 40.0, 2.0], vec![0.8, 0.4, 0.2]),
    //     (vec![50.0, 1.80, 25.0, 4.0], vec![0.4, 0.8, 0.8]),
    // ];

    let dados_treinamento: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![70.0, 1.75, 30.0, 3.0], vec![0.6, 0.6, 0.6]), // Peso normal, atividade moderada
        (vec![90.0, 1.65, 40.0, 2.0], vec![0.8, 0.4, 0.2]), // Sobrepeso, atividade leve
        (vec![50.0, 1.80, 25.0, 4.0], vec![0.4, 0.8, 0.8]), // Abaixo do peso, atividade intensa
        (vec![110.0, 1.70, 50.0, 1.0], vec![0.9, 0.2, 0.1]), // Obesidade grau I, sedentário
        (vec![60.0, 1.60, 35.0, 5.0], vec![0.5, 0.9, 0.9]), // Peso normal, extremamente ativo
        (vec![85.0, 1.85, 28.0, 2.0], vec![0.7, 0.3, 0.3]), // Sobrepeso, levemente ativo
        (vec![45.0, 1.55, 22.0, 3.0], vec![0.3, 0.7, 0.7])  // Abaixo do peso, moderadamente ativo
    ];


    let dados_normalizados: Vec<(Vec<f32>, Vec<f32>)> = dados_treinamento.iter().map(|d| (normalizar_dados(&vec![d.0.clone()])[0].to_vec(), d.1.clone())).collect();
    let taxa_aprendizagem = 0.1;
    let epocas = 1000;
    let (pesos_oculta, biases_oculta, pesos_saida, biases_saida) = treinar_rede(&dados_normalizados, taxa_aprendizagem, epocas);

    loop {
        let entradas_usuario = obter_entradas();
        let saida_rede = calcular_saida(&entradas_usuario, &(pesos_oculta.clone(), pesos_saida.clone()), &(biases_oculta.clone(), biases_saida.clone()));
        let (imc_msg, recomendacao_msg1, recomendacao_msg2) = interpretar_saida(&saida_rede);

        println!("---------------------");
        println!("IMC estimado: {} - {}", saida_rede[0], imc_msg);
        println!("---------------------");
        println!("Recomendações de atividade física:");
        println!("1. {}", recomendacao_msg1);
        println!("2. {}", recomendacao_msg2);
    }
}
