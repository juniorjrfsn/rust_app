use rand::prelude::*; // Importa tudo necessário
use rand::seq::SliceRandom; // Garante que o `choose` funcione

fn process_words_matrix(matrix: Vec<Vec<String>>) -> Vec<String> {
    let mut rng = rand::thread_rng(); // Se der erro, tente substituir por `rand::rng()`
    matrix
        .iter()
        .filter_map(|row| row.choose(&mut rng).cloned())
        .collect()
}

fn main() {
    let words_matrix = vec![
        vec!["gato".to_string(), "cachorro".to_string(), "papagaio".to_string()],
        vec!["vermelho".to_string(), "azul".to_string(), "verde".to_string()],
        vec!["rápido".to_string(), "lento".to_string(), "médio".to_string()],
    ];

    let result = process_words_matrix(words_matrix);
    println!("Palavras selecionadas: {:?}", result);
}
