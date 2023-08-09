pub mod ctrl_person {
	extern crate rusqlite; 
	use rusqlite::{Connection, Result};

	#[derive(Debug)]
	struct Pessoa {
		id: i32,
		name: String,
		data: Option<Vec<u8>>,
	}
	/*
	impl Pessoa {
		fn id(&self) -> &i32 {
			&self.id
		}
		fn name(&self) -> &String {
			&self.name
		}
		fn data(&self) -> &Option<Vec<u8>> {
			&self.data
		}
	}
	*/
	pub fn registrar() -> Result<()> {
		let conn = Connection::open("my_db.db")?;

		let me = Pessoa {
			id: 0,
			name: "Steven".to_string(),
			data: None, // None
		};
		conn.execute( "INSERT INTO person (name, data) VALUES (?1, ?2)", (&me.name, &me.data), )?;
		Ok(())
	}

	pub fn get_persons() -> Result<()> {
		let conn = Connection::open("my_db.db")?;
 
		println!(" ======================   " );

		let mut stmt = conn.prepare("SELECT MAX(id) AS id, name, data FROM person GROUP BY name, data ")?;
		let person_iter = stmt.query_map([], |row| {
			Ok(Pessoa {
				id: row.get(0)?,
				name: row.get(1)?,
				data: row.get(2)?,
			})
		})?;
		for person in person_iter.into_iter() {
			//println!("Found person {:?}", person.unwrap());
			let pessoa = person.unwrap();
			//let dados = String::from_utf8(pessoa.data).expect("Found invalid UTF-8");
			//let dados = pessoa.data;

			//  String::from_utf8(dados ).expect("Found invalid UTF-8"));
			// let vec: Vec<u8> = vec![0xa8, 0x3c, 0x09];

			//let op: Option<Vec<u8>> =    Option<vec> ;
			//let v = [Ok(2), Ok(4), Err("err!"), Ok(8)];
			// let res: Result<Vec<_>, &str> = v.into_iter().collect();
			//let dados : String = if pessoa.data.is_some(){ String::from_utf8(  pessoa.data.unwrap()  ).unwrap()  } else {   String::new() };

			println!("Pessoa:{} - nome:{:?} : dados:{} ", pessoa.id, pessoa.name, if pessoa.data.is_some(){ String::from_utf8(  pessoa.data.unwrap()  ).unwrap()  } else {   String::new() } ); 
			/*
				if pessoa.data.is_some() {
					println!("Pessoa:{} - nome:{} : dados:{:?} ", pessoa.id, pessoa.name, pessoa.data ); 
				} else {
					println!("Pessoa:{} - nome:{} : dados:{} ", pessoa.id, pessoa.name, "" );  
				}
			*/
		}
		
 

		Ok(())

	}


}