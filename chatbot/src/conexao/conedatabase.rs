pub mod conectdatabase {
	use rusqlite::{Connection, Result};

	pub fn create_database()  -> Result<()> {
		let _path = "consciencia/estimulo.db";
		// let path = "./cats.db";
		let db = Connection::open(_path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}
}