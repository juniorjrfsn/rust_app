mod conexao;
use crate::conexao::cone::conect;

fn main(){
    let _fn1 = conect::create_database();
    let _fn2 = conect::create();
    let _fn3 = conect::registrar();
    let _fn4 = conect::get_cats();
}