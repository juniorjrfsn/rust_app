use rand::Rng;
fn main() {

    println!("");
    println!("-------------------------------------------");
    println!("Hello, world!");
    println!("-------------------------------------------");
    println!("");
    // Crie um gerador de números aleatórios.
    let mut rng = rand::thread_rng();

    // Crie uma string vazia para armazenar a cadeia de caracteres.
    let mut string = String::new();

    // Gere 6 dígitos aleatórios.
    let numbers = vec![1, 2, 3, 4, 5, 6];
    
    for number in numbers.iter() {
       // println!("Number: {}", number);
       let mut cont = 0;
       for _ in 0..6 {
            for _ in 0..2 {
                let digit = rng.gen_range(0..10);
                string.push_str(&digit.to_string());
                // Imprima a string gerada.
                // println!("{}", string);
            }
            cont = cont+1;
            if cont < 6 {
                string.push_str(&"-")
            }
        }
        println!("{}", string);
        string = String::new();
    }
}
