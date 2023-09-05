use std::io;
fn main() {
    let palavras = vec!["casa", "carro", "árvore", "cidade", "cão", "gato", "flor", "mar", "lua", "sol"];

    let palavras_com_a: Vec<&str> = palavras
        .iter()
        // .filter(|&&palavra| palavra.contains('a'))
        .filter(|&&palavra| {
            palavra.contains('a')
        })
        .cloned()
        .collect();
    println!("{:?}", palavras_com_a);

    let mut numbers: Vec<i32> = Vec::new();
    loop {
        println!("Escolha uma opcao:");
        println!("1. Adicionar numero a lista");
        println!("2. Mostrar lista");
        println!("3. Sair");

        let mut choice = String::new();
        io::stdin()
            .read_line(&mut choice)
            .expect("Falha ao ler a linha");
        match choice.trim() {
            "1" => {
                println!("Digite um numero:");
                let mut input = String::new();
                io::stdin()
                    .read_line(&mut input)
                    .expect("Falha ao ler a linha");
                let num: i32 = match input.trim().parse() {
                    Ok(num) => num,
                    Err(_) => {
                        println!("Entrada invalida. Por favor, insira um numero valido.");
                        continue;
                    }
                };
                numbers.push(num);
                println!("Numero {} adicionado a lista.", num);
            }
            "2" => {
                println!("Lista de numeros:");
                for (index, num) in numbers.iter().enumerate() {
                    println!("{}: {}", index + 1, num);
                }
            }
            "3" => {
                println!("Saindo da aplicacao.");
                break;
            }
            _ => {
                println!("Opcao invalida. Por favor, escolha uma opcao valida.");
            }
        }
    }
}