pub mod  conect {
	extern crate rusqlite;
	//use rusqlite::{Connection, Result};
	use rusqlite::{Connection, Result};
 

	pub fn createDatabase()  -> Result<()> {
		let path = "./my_db.db";
		let db = Connection::open(path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}

	pub fn create()  -> Result<()> {
		let conn = Connection::open("my_db.db")?;

		conn.execute_batch(
			"BEGIN;
			CREATE TABLE foo(x INTEGER);
			CREATE TABLE bar(y TEXT);
			COMMIT;",
		);
 
		conn.execute_batch(
			"create table if not exists cat_colors (
			id integer primary key,
			name text not null unique
			)",
		)?;
		conn.execute_batch(
			"create table if not exists cats (
			id integer primary key,
			name text not null,
			color_id integer not null references cat_colors(id)
			)",
		)?;

		Ok(())
	}

}