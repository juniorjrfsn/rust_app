use std::io::{BufRead, BufReader};
use std::fs::File;

fn main() {
    let arquivo = BufReader::new(File::open("arquivosplit.toml").unwrap());

    for linha in arquivo.lines() {
        let linha = linha.unwrap();
        println!("{}", linha);
    }
}