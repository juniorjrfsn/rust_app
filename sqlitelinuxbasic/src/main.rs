mod exemplos;
use crate::exemplos::codigos::codes;

mod conexao;

use crate::conexao::conedatabase::conectdatabase;
use crate::conexao::control_cat::ctrl_cat;
use crate::conexao::control_person::ctrl_person;
use crate::conexao::migrationdb::migrationtable;

fn main() {

    // Gera os codigos string
    println!("-------------------------------------------");
    let _fn1: () = codes::get_codes_string(Some(false));
    println!("-------------------------------------------");

    // Gera os codigos
    let _fn2: () = codes::get_codes(Some(false));
    println!("-------------------------------------------");

    // Cria o banco de dados
    let _fn3: Result<(), rusqlite::Error> = conectdatabase::create_database();
    println!("-------------------------------------------");

    // Cria as tabelas
    let _fn4: Result<(), rusqlite::Error> = migrationtable::migration_create_table();
    println!("-------------------------------------------");

    // Registra um gato
    let c: bool = true;
    let _fn5: Result<(), rusqlite::Error> = ctrl_cat::registrar(c);
    println!("-------------------------------------------");

    // Busca todos os gatos
    let _fn6: Result<(), rusqlite::Error> = ctrl_cat::get_cats();
    println!("-------------------------------------------");

    // Registra uma pessoa
    let p: bool = true;
    let _fn7: Result<(), rusqlite::Error> = ctrl_person::registrar(p);
    println!("-------------------------------------------");

    // Atualiza uma pessoa
    let hello: String = String::from("Hello, world! agora siim").to_owned();
    let vec: Vec<u8> = hello.into_bytes();  // let vec: Vec<u8> = vec![0xaa, 0xfc, 0x09, 0x09];
    let op: Option<Vec<u8>> = Some(vec);
    let _fn8 = ctrl_person::update_row(1, "John", op, false);
    println!("-------------------------------------------");

    // Busca todas as pessoas
    let _fn6: Result<(), rusqlite::Error> = ctrl_person::get_persons();
    println!("-------------------------------------------");
}
