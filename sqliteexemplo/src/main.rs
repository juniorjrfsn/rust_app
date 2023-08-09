mod conexao;
use crate::conexao::cone::conect;
use crate::conexao::conedatabase::conectdatabase;
use crate::conexao::migrationdb::migrationtable;
use crate::conexao::control_cat::ctrl_cat;
use crate::conexao::control_person::ctrl_person;

fn main(){
    // println!("Connection : {:?}",   if _fn1.Ok(10) { _fn1  } else {   Err(10) } );
    //let _fn1 = conect::create_database();
    let _fn1 = conectdatabase::create_database();
    //let _fn2 = conect::create_table();
    let _fn2 = migrationtable::migration_create_table();
    let _fn3 = conect::get_codes();
    let _fn3 = ctrl_cat::registrar();
    let _fn4 = ctrl_cat::get_cats();
    let _fn3 = ctrl_person::registrar();
    let _fn5 = ctrl_person::get_persons();
}