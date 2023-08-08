pub mod  conect {
	extern crate rusqlite;
	//use rusqlite::{Connection, Result};
	use rusqlite::{Connection, Result};
	use std::collections::HashMap;

	#[derive(Debug)]
	struct Cat {
		name: String,
		color: String,
	}



	pub fn createDatabase()  -> Result<()> {
		let path = "./my_db.db";
		let path = "./cats.db";
		let db = Connection::open(path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}

	// not working
	pub fn create()  -> Result<()> {
		let conn = Connection::open("my_db.db")?;

		conn.execute(
			"BEGIN;
			CREATE TABLE foo(x INTEGER);
			CREATE TABLE bar(y TEXT);
			COMMIT;",[],
		);
 
		conn.execute(
			"create table if not exists cat_colors (
			id integer primary key,
			name text not null unique
			)",[],
		)?;
		conn.execute(
			"create table if not exists cats (
			id integer primary key,
			name text not null,
			color_id integer not null references cat_colors(id)
			)",[],
		)?; 

		Ok(())
	}

	pub fn registrar() -> Result<()> {
		let conn = Connection::open("cats.db")?;

		let mut cat_colors = HashMap::new();
		cat_colors.insert(String::from("Blue"), vec!["Tigger", "Sammy"]);
		cat_colors.insert(String::from("Black"), vec!["Oreo", "Biscuit"]);
		cat_colors.insert(String::from("white"), vec!["branco", "rajado"]);

		for (color, catnames) in &cat_colors {
			conn.execute( "INSERT INTO cat_colors (name) values (?1)", &[&color.to_string()], )?;
			let last_id: String = conn.last_insert_rowid().to_string();

			for cat in catnames {
				conn.execute( "INSERT INTO cats (name, color_id) values (?1, ?2)", &[&cat.to_string(), &last_id], )?;
			}
		}

		Ok(())
	}

	pub fn getCats() -> Result<()> {
		let conn = Connection::open("cats.db")?;

		let mut stmt = conn.prepare( "SELECT c.name, cc.name from cats c INNER JOIN cat_colors cc ON cc.id = c.color_id;", )?;

		let cats = stmt.query_map([], |row| {
			Ok(Cat {
				name: row.get(0)?,
				color: row.get(1)?,
			})
		})?;

		for cat in cats {
			println!("Gato {:?}   ", cat.unwrap());
		}

		Ok(())
	}
}