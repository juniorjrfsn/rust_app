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

fn main() {
    println!("Hello, world!");
    let _fn1 = conectdatabase::create_database();
    println!("-------------------------------------------" );
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Falha ao ler a linha");
    // let num: i32 = match input.trim().parse() { Ok(num) => num, Err(_) => 0, };
    

    match (match input.trim().parse() { Ok(num) => num, Err(_) => 0, }) == 0 {
        true => {
            let _fn1: () = codes::get_codes_string(Some(false));
        }
        false => {
            let _fn1: () = codes::get_codes_string(Some(true));
        }
    }
}


// cd estimulo
// cargo run