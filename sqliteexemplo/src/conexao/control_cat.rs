pub mod ctrl_cat {
	extern crate rusqlite; 
	use rusqlite::{Connection, Result};
	use std::collections::HashMap;

 	#[derive(Debug)]
	struct Cat {
		name: String,
		color: String,
	}

	impl Cat {
		fn name(&self) -> &String {
			&self.name
		}
		fn color(&self) -> &String {
			&self.color
		}
	}
 
	pub fn registrar(reg: bool) -> Result<()> {
		if reg  {
			let conn = Connection::open("my_db.db")?;
			let mut cat_colors = HashMap::new();

			// a tabela no banco de dados possui restrição contra multiplicidade de nomes
			cat_colors.insert(String::from("Blue"),		vec!["Tigger",	"Sammy"]	);
			cat_colors.insert(String::from("Black"),	vec!["Oreo",	"Biscuit"]	);
			cat_colors.insert(String::from("white"),	vec!["branco",	"rajado"]	);
			cat_colors.insert(String::from("Yellow"),	vec!["amarelo",	"caramelo"]	);
			cat_colors.insert(String::from("Marron"),	vec!["Nego",	"Nega"]	);

			for (color, catnames) in &cat_colors {
				conn.execute( "INSERT INTO cat_colors (name) values (?1)", &[&color], )?;
				let last_id: String = conn.last_insert_rowid().to_string();
				
				for cat in catnames {
					conn.execute( "INSERT INTO cats (name, color_id) values (?1, ?2)", &[&cat.to_string(), &last_id], )?;
				}
			}
		} else {

		}
		Ok(())
	}

	pub fn get_cats() -> Result<()> {
		let conn = Connection::open("my_db.db")?;
		/*
			for (id, cat) in cats.into_iter().enumerate()  {
				//println!("{} - {}", row.name, row.color);
				println!("Item {:?} ", cat.unwrap() );


				let obj = HashMap::from( cat );
				for prop in obj.keys() {
					println!("{}: {}", prop, obj.get(prop).unwrap());
				}


				/*let mapa = HashMap::from( cat );
				for (key, val) in mapa.iter() {
					println!("key: {key} val: {val}");
				}*/
			}
		*/
 
		let mut stmt = conn.prepare( "SELECT c.name, cc.name from cats c INNER JOIN cat_colors cc ON cc.id = c.color_id;", )?;
		let cats = stmt.query_map([], |row| { 
			Ok(Cat {
				name: row.get(0)?,
				color: row.get(1)?,
			}) 
		})?;
		for cat in cats.into_iter()  {
			let gato = cat.unwrap();
			println!("Gato {} de cor {:?}  ", gato.name(), gato.color() );
		}
		

		Ok(())
	}

}