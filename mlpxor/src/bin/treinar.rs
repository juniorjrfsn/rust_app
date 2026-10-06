use mlpxor::RedeNeural;

fn main() {
    println!("========================================");
    println!("     TREINAMENTO DA REDE NEURAL XOR");
    println!("========================================");

    println!("\nArquitetura:");
    println!("  Entrada  : 2 neurônios");
    println!("  Oculta   : 3 neurônios - tanh");
    println!("  Saída    : 1 neurônio  - sigmoid");

    println!("\nConfiguração:");
    println!("  Taxa de aprendizado : 0.1");
    println!("  Épocas              : 500000");

    println!("\nTabela XOR:");
    println!("  [0, 0] -> [0]");
    println!("  [0, 1] -> [1]");
    println!("  [1, 0] -> [1]");
    println!("  [1, 1] -> [0]");

    let mut rede = RedeNeural::nova(&[2, 3, 1]);

    let tabela_xor = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    println!("\n----------------------------------------");
    println!("Iniciando treinamento...");
    println!("----------------------------------------");

    rede.treinar(
        &tabela_xor,
        0.5,
        50,
    );

    rede.salvar_modelo("pesos_xor.json");

    println!("Modelo salvo em: pesos_xor.json");
}