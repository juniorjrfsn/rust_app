
mod conexao;
use crate::conexao::conedatabase::conectdatabase;
use crate::conexao::migrationdb::migrationtable;
use crate::conexao::control_cat::ctrl_cat;
use crate::conexao::control_person::ctrl_person;
use crate::conexao::codigos::codes;

fn main() {
    let _fn1 = conectdatabase::create_database();
    println!("-------------------------------------------" );

    let _fn2 = migrationtable::migration_create_table();
    println!("-------------------------------------------" );

    let _fn3 = codes::get_codes();
    println!("-------------------------------------------" );

    let c  = true;
    let _fn3 = ctrl_cat::registrar(c);
    println!("-------------------------------------------" );

    let _fn4 = ctrl_cat::get_cats();
    println!("-------------------------------------------" );

    let p  = true;
    let _fn3 = ctrl_person::registrar(p);
    println!("-------------------------------------------" );

    let hello = String::from("Hello, world! agora siim").to_owned();
    let vec:Vec<u8> = hello.into_bytes();

    // let vec: Vec<u8> = vec![0xaa, 0xfc, 0x09, 0x09];
    let op: Option<Vec<u8>> = Some(vec) ;
    let _fn4 = ctrl_person::update_row(1, "John", op, false);
    println!("-------------------------------------------" );

    let _fn5 = ctrl_person::get_persons();
    println!("-------------------------------------------" );
 
}
