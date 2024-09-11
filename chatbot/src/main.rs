use std::fs;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::prelude::*;
#[cfg(target_family = "unix")]
use std::os::unix;
#[cfg(target_family = "windows")]
use std::os::windows;
use std::path::Path;

mod exemplos;
use crate::exemplos::codigos::codes;

mod conexao;
use crate::conexao::conedatabase::conectdatabase;

mod dao;
use crate::dao::ai::chatbot;

pub(crate) fn main(){
    println!("Hello, world!");
    let _fn1 = conectdatabase::create_database();
    loop {
        println!("Digite S ou 1 par iniciar uma conversa:");

        let mut escolha = String::new();
        // io::stdin().read_line(&mut escolha).expect("Failed to read line");

        // Extraímos o primeiro caractere da string
        // trim():remove espaços em branco, chars():transforma a string em um iterador de caracteres, next():retorna o primeiro caracterer do iterador - expect:trata os erros
        // let character = escolha.trim().chars().next().expect("Please enter a character");

        let mut character: char = 'N';
        match io::stdin().read_line(&mut escolha) {
            Ok(_) => {
                // Leitura bem-sucedida
                character = match escolha.trim().chars().next() {
                    Some(c) if c.is_alphabetic() && (c == 'S' || c == 's' || c == '1') => c,
                    _ => {
                        println!("Caractere inválido. Considerando 'S' como padrão.");
                        'N'
                    }
                };
                // println!("O caractere escolhido foi: {}", character);
            }
            Err(error) => {
                // Tratamento de erro
                println!("Ocorreu um erro durante a leitura: {}", error);
            }
        }

        // Verificamos se o caractere é 'S' ou '1'
        let var_name = if character == 'S' || character == 's' || character == '1'   {
            // println!("Você digitou {}.", character);
            println!("O que você gostaria de saber?" );
            loop {
                let mut per = String::new();
                io::stdin().read_line(&mut per).expect("Falha ao ler a linha");
                // let num: i32 = match input.trim().parse() { Ok(num) => num, Err(_) => 0, };

                let _fn_: Result<(), rusqlite::Error> = chatbot::perguntar(per.clone());
                match (match per.trim().parse() { Ok(num) => num, Err(_) => 0, }) == 0 {
                    true => {
                    let _fn1: () = codes::get_codes_string(Some(false));
                    }
                    false => {
                    let _fn1: () = codes::get_codes_string(Some(true));
                    }
                }
                println!("-------------------------------------------" );
            }
        } else {
            println!("Você optou por treinar o chatbot. {}", character);

            println!("Escreva a resposta de forma resumida e clara." );


        };
        var_name
    }
}

// cd estimulo
// cargo run
// cargo run --bin estimulo
// cargo.exe "run", "--package", "estimulo", "--bin", "estimulo"