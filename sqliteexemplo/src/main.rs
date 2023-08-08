use rusqlite::{Connection, Result};
 
mod conexao;
use crate::conexao::cone::conect;

fn main()  -> Result<()> {
    conect::createDatabase();
    // conect::create();
    
    Ok(())
}