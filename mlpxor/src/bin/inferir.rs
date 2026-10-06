use mlpxor::RedeNeural;

fn main() {

    println!("=== INICIANDO INSTRUÇÃO DE INFERÊNCIA ===");

    // Carrega a rede treinada sem precisar calcular backpropagation
    let rede = RedeNeural::carregar_modelo("pesos_xor.json");

    let testes = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    println!("\nResultados da Inferência:");
    println!("---------------------------------------");
    for entrada in testes {
        println!("\n---------------------------------------");

        let saida = rede.inferir_detalhado(&entrada);

        let classe = if saida > 0.5 { 1 } else { 0 };

        println!("  Classe: {}", classe);
    }
    println!("---------------------------------------");
}