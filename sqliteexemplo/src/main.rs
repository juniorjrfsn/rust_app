mod conexao;
use crate::conexao::conedatabase::conectdatabase;
use crate::conexao::migrationdb::migrationtable;
use crate::conexao::control_cat::ctrl_cat;
use crate::conexao::control_person::ctrl_person;

use crate::conexao::codigos::codes;

fn main(){
    // println!("Connection : {:?}",   if _fn1.Ok(10) { _fn1  } else {   Err(10) } );
 
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
    let _fn5 = ctrl_person::get_persons();
    println!("-------------------------------------------" );
}