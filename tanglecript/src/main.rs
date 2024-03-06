mod aposta;
use crate::aposta::gerador::ger;

fn main() {

    println!("");
    println!("-------------------------------------------");
    println!("Hello, world!");
    println!("-------------------------------------------");

    let mut apostas: Vec<String> = vec![];
    for _ in 0..6 {
        let mut apo  = ger::gera_aposta();
        // println!("{}", ger::gera_aposta());
        // apostas.push(ger::gera_aposta().to_string());
        while apostas.iter().any(|x| x == &apo) {
            apo = ger::gera_aposta();
        }
        apostas.push(apo);
    }
    for apostinha in apostas {
        println!("{}", apostinha);
    }
    println!("-------------------------------------------");

    // Gere 6 dígitos aleatórios.
    let numbers = vec![1, 2, 3, 4, 5, 6];
    for number in numbers.iter() {
        println!("Number: {}", number);
    }
       // string = String::new();

}
