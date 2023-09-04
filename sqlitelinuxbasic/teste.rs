fn main() {
    let palavras = vec!["casa", "carro", "árvore", "cidade", "cão", "gato", "flor", "mar", "lua", "sol"];

    let palavras_com_a: Vec<&str> = palavras
        .iter()
        .filter(|&&palavra| palavra.contains('a'))
        .cloned()
        .collect();
    
    println!("{:?}", palavras_com_a);

  

}