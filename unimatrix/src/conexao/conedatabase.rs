pub mod conectdatabase {
	use rusqlite::{Connection, Result};

	pub fn create_database()  -> Result<()> {
		let _path = "unimatrix.db";
		// let path = "./cats.db";
		let db = Connection::open(_path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}
}