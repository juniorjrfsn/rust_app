use std::fs::File;
use std::io::{BufRead, BufReader};
fn main() {
    let file = File::open("files/estados.json").unwrap();
    let reader = BufReader::new(file);

    let palavra = "sigla";

    for line in reader.lines() {
        let line = line.unwrap();

        if line.contains(palavra) {
            println!("A palavra '{}' existe no arquivo.", palavra);
            break;
        }
    }
    if !palavra.contains(palavra) {
        println!("A palavra '{}' nao existe no arquivo.", palavra);
    }
}