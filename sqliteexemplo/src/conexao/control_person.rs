pub mod ctrl_person {
	extern crate rusqlite;
	use std::str;
	use std::error::Error;
	use std::io::prelude::*;
	use std::process::{Command, Stdio};
	use rusqlite::{Connection, Result};

	#[derive(Debug)]
	struct Pessoa {
		id	: i32,
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

	pub fn registrar(reg: bool ) -> Result<()> {
		if reg {
			let conn = Connection::open("my_db.db")?;

			let hello = String::from("Mandando Blob pro banco").to_owned();
			let vec:Vec<u8> = hello.into_bytes();
				// let vec: Vec<u8> = vec![0xaa, 0xfc, 0x09, 0x09];
			let op: Option<Vec<u8>> = Some(vec) ;
			let me = Pessoa {
				id: 1,
				name: "John".to_string(),
				data: op, // None
			};
			conn.execute( "INSERT INTO person (name, data) VALUES (?1, ?2)", (&me.name, &me.data), )?;

			let hello2 = String::from("Mandando Blob pro banco").to_owned();
			let vec2:Vec<u8> = hello2.into_bytes();
				// let vec: Vec<u8> = vec![0xaa, 0xfc, 0x09, 0x09];
			let op2: Option<Vec<u8>> = Some(vec2) ;
			let me2 = Pessoa {
				id: 2,
				name: "Steven".to_string(),
				data: op2, // None
			};
			conn.execute( "INSERT INTO person (name, data) VALUES (?1, ?2)", (&me2.name, &me2.data), )?;

		} else { }
		Ok(())
	}

	/*
	pub fn  print_dados_update(id: i32, name: &str, data: Option<Vec<u8>>, _reg: bool) -> Result<()>{
		println!("Pessoa:{} - nome:{:?} : dados:{:?} ", id , name, data  );
		Ok(())
	}
	*/

	pub fn update_row(id: i32, name: &str, data: Option<Vec<u8>>, reg: bool) -> Result<(), Box<dyn Error>> {

		// let vec2: Option<Vec<u8>> = Some(Vec::new());
		// let vec2: Option<Vec<u8>> = Some(Vec::new());


		// let opt: Option<Vec<u8>> = Some(data);
		// let vec: Vec<u8> = vec![0xa8, 0x3c, 0x09];
	    // let vec2:Option<Vec<u8>>  = data ;
	    // vec2.copy_from(data);
	    // Print the vector
  		// let  dados:  Vec<u8> = data.unwrap().to_string().as_str(), ;
  		// println!("{:?}", vec);
  		// let  dados:  &str = String::from_utf8( data.unwrap()  ).unwrap().to_string().as_str().as_ref();
  		let dados:  &Vec<u8> = data.as_ref().unwrap();
  		//let dados1: &str = str::from_utf8(dados).unwrap();
		let mut tmp = String::new();
		tmp.push_str(&String::from_utf8_lossy(dados));

  		// let dados1 : &str = dados.as_ref();
  		let conn = Connection::open("my_db.db")?;
		if reg {
			// Update a row

			//  conn.execute("UPDATE person SET name = ?, data = ? WHERE id = ?", &[name, String::from_utf8( data.unwrap()  ).unwrap().to_string().as_str() ,  id.to_string().as_str()] )?;
			conn.execute("UPDATE person SET name = ?, data = ? WHERE id = ? ", &[name, tmp.as_str() ,  id.to_string().as_str()] )?;
			// conn.execute("UPDATE person SET name = ?, data = ? WHERE id = ?", &[name, dados.as_ref() ] )?;
			// println!("Pessoa:{} - nome:{:?} : dados:{} ", id.to_string(), name, if data.is_some(){ String::from_utf8(  data.unwrap()  ).unwrap()  } else {   String::new() } ); 

		} else {
			// println!("Pessoa:{} - nome:{:?} : dados:{:?} ", id , name, vec2  );
		}
		// let mut to_child = data.as_ref().unwrap();
		// println!("Pessoa:{} - nome:{:?} : dados:{:?} ", id , name, vec2  );
		println!("Pessoa:{} - nome:{:?} : dados:{:?} ", id , name, tmp.as_str() );

		// let _fn_exec = print_dados_update(id, name, data.as_ref(), reg);
		// conn.wait()?;
		Ok(())
	}

	/*
	pub fn update_ver_row( id: i32, name: &str, data: Option<Vec<u8>>, reg: bool)  {
		let mut to_child = data.as_ref().unwrap();
		if reg {
			//let _fn_exec = update_row(id, name, data, reg);
		} else {
		}
		println!("Pessoa:{} - nome:{:?} : dados:{:?} ", id , name, data.unwrap()  );
	}
	*/

	pub fn get_persons() -> Result<()> {
		let conn = Connection::open("my_db.db")?;

		let mut stmt = conn.prepare("SELECT MAX(id) AS id, name, data FROM person GROUP BY name, data ")?;
		let person_iter = stmt.query_map([], |row| {
			Ok(Pessoa {
				id		: row.get(0)?,
				name	: row.get(1)?,
				data	: row.get(2)?,
			})
		})?;

		for person in person_iter  {
			// println!("Found person {:?}", person.unwrap());
			let pessoa = person.unwrap();
			// let dados = String::from_utf8(pessoa.data).expect("Found invalid UTF-8");
			// let dados = pessoa.data;

			//  String::from_utf8(dados ).expect("Found invalid UTF-8"));
			// let vec: Vec<u8> = vec![0xa8, 0x3c, 0x09];

			// let op: Option<Vec<u8>> =    Option<vec> ;
			// let v = [Ok(2), Ok(4), Err("err!"), Ok(8)];
			// let res: Result<Vec<_>, &str> = v.into_iter().collect();
			// let dados : String = if pessoa.data.is_some(){ String::from_utf8(  pessoa.data.unwrap()  ).unwrap()  } else {   String::new() };

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