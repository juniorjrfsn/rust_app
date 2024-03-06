mod aposta;
use crate::aposta::gerador::ger;

fn main() {

    println!("");
    println!("-------------------------------------------");
    println!("Hello, world!");
    println!("-------------------------------------------");

    for _ in 0..6 {
        println!("{}", ger::gera_aposta());
    }
    println!("-------------------------------------------");

    // Gere 6 dígitos aleatórios.
    let numbers = vec![1, 2, 3, 4, 5, 6];
    for number in numbers.iter() {
        println!("Number: {}", number);
    }
       // string = String::new();

}
