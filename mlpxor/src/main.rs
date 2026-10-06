// mlp_xor/src/main.rs

use mlpxor::RedeNeural;

fn main() {
    let caminho_modelo = "pesos_xor.json";

    println!("--- 1. FASE DE TREINAMENTO ---");
    let mut rede = RedeNeural::nova(&[2, 3, 1]);
    let tabela_xor = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    rede.treinar(&tabela_xor, 0.1, 50000);
    rede.salvar_modelo(caminho_modelo);

    println!("\n--- 2. FASE DE INFERÊNCIA ---");
    let rede_carregada = RedeNeural::carregar_modelo(caminho_modelo);

    let testes = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    for entrada in testes {
        let valor_bruto = rede_carregada.inferir(&entrada);
        let valor_binario = if valor_bruto > 0.5 { 1 } else { 0 };

        println!(
            "Entrada: {:?} | Saída Bruta: {:.4} | Previsão XOR: {}",
            entrada, valor_bruto, valor_binario
        );
    }
}


// cd mlpxor
// cargo run
// cargo run --bin treinar
// cargo run --bin inferir  

// cargo run --bin mlpxor
// cargo.exe "run", "--package", "mlpxor", "--bin", "mlpxor"
// cargo.exe "run" "--package" "mlpxor" "--bin" "mlpxor"
