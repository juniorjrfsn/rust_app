pub mod  conect {
	use rusqlite::{Connection, Result};

	pub fn createDatabase()  -> Result<()> {
		let path = "./my_db.db";
		let db = Connection::open(path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}

}