use std::fs;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::prelude::*;
#[cfg(target_family = "unix")]
use std::os::unix;
#[cfg(target_family = "windows")]
use std::os::windows;
use std::path::Path;

mod conexao;
use crate::conexao::conedatabase::conectdatabase;
fn main() {
    println!("Hello, world!");
    let _fn1 = conectdatabase::create_database();
    println!("-------------------------------------------" );
}


// cd estimulo
// cargo run