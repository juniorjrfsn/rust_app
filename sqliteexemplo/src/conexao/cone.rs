pub mod  conect {
	extern crate rusqlite;
	//use rusqlite::{Connection, Result};
	use rusqlite::{Connection, Result};
	use std::collections::HashMap;

	// ========== não usado ========
	use std::ops::{Index, IndexMut}; 
	pub struct Vector3d<T> {
	    pub x: T,
	    pub y: T,
	    pub z: T,
	}
	impl<T> Index<usize> for Vector3d<T> {
	    type Output = T;
	    fn index(&self, index: usize) -> &T {
	        match index {
	            0 => &self.x,
	            1 => &self.y,
	            2 => &self.z,
	            n => panic!("Invalid Vector3d index: {}", n)
	        }
	    }
	}
	impl<T> IndexMut<usize> for Vector3d<T> {
	    fn index_mut(&mut self, index: usize) -> &mut T {
	        match index {
	            0 => &mut self.x,
	            1 => &mut self.y,
	            2 => &mut self.z,
	            n => panic!("Invalid Vector3d index: {}", n)
	        }
	    }
	}
	// ========== não usado ========


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
	pub fn create_database()  -> Result<()> {
		let _path = "my_db.db";
		//let path = "./cats.db";
		let db = Connection::open(_path)?;
		// Use the database somehow...
		println!("BASE DE DADOS OPERANDO : {}", db.is_autocommit());
		Ok(())
	}

	#[derive(Debug)]
	struct Pessoa {
		id: i32,
		name: String,
		data: Option<Vec<u8>>,
	}

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

	// not working
	pub fn create_table()  -> Result<()> {
		let conn = Connection::open("my_db.db")?;

		let _r1 = conn.execute(
			"BEGIN;
			CREATE TABLE foo(x INTEGER);
			CREATE TABLE bar(y TEXT);
			COMMIT;",[],
		);
 
		let _r2 = conn.execute(
			"create table if not exists cat_colors (
			id integer primary key,
			name text not null unique
			)",[],
		)?;
		let _r3 = conn.execute(
			"create table if not exists cats (
			id integer primary key,
			name text not null,
			color_id integer not null references cat_colors(id)
			)",[],
		)?; 
		let _r4 = conn.execute(
			"CREATE TABLE person (
				id INTEGER PRIMARY KEY,
				name TEXT NOT NULL,
				data BLOB
			)",[],
		)?; 

		Ok(())
	}

	pub fn registrar() -> Result<()> {
		let conn = Connection::open("my_db.db")?;

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
 
		let me = Pessoa {
			id: 0,
			name: "Steven".to_string(),
			data: None, // None
		};
		conn.execute( "INSERT INTO person (name, data) VALUES (?1, ?2)", (&me.name, &me.data), )?;
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
		println!(" ======================   " );
		let mut state_codes: HashMap<&str, &str> = HashMap::new();
		state_codes.insert("NV", "Nevada");
		state_codes.insert("NY", "New York");

		for (key, val) in state_codes.iter() {
			println!("key: {key} val: {val}");
		}
		println!(" ======================   " );
		let map = HashMap::from([
			("a", 1),
			("b", 2),
			("c", 3),
		]);

		for (key, val) in map.iter() {
			println!("key: {key} val: {val}");
		}
		println!(" ======================   " );
		let obj = HashMap::from([
			("key1", "value1"),
			("key2", "value2")
		]);
		for prop in obj.keys() {
			println!("{}: {}", prop, obj.get(prop).unwrap());
		} 
		println!(" ======================   " );
		let numbers = [1, 2, 3, 4, 5];
		for number in numbers {
			println!("{}", number);
		} 
		println!(" ======================   " );
		let numbers = [1, 2, 3, 4, 5];
		let first_even = numbers.iter().find(|x| *x % 2 == 0);
		println!("{:?}", first_even.unwrap());
		println!(" ======================   " );
		let numbers = [1, 2, 3];
		numbers.iter().for_each(|x| println!("{}", x));
		println!(" ======================   " );
		let names = ["Sam", "Janet", "Hunter"];
		let csv = names.join(", ");
		println!("{}", csv);
		println!(" ======================   " );
		let my_array1 = [1, 2, 3, 4, 5];
		let mut index = 0; 
		while index < my_array1.len() {
			println!("{}", my_array1[index]);
			index += 1;
		}
		println!(" ======================   " );
		let my_array2 = [1, 2, 3, 4, 5];
		for item in my_array2.iter() {
			println!("{}", item);
		}
		println!(" ======================   " );
		let my_array = [1, 2, 3, 4, 5];
		for (index, item) in my_array.iter().enumerate() {
			println!("{}: {}", index, item);
		}
		println!(" ======================   " );


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

		println!(" ======================   " );

		let mut stmt = conn.prepare("SELECT id, name, data FROM person")?;
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
			let dados = if pessoa.data.is_some(){ pessoa.data } else { ""};
			println!("Pessoa:{} - nome:{} : dados:{:?} ", pessoa.id, pessoa.name, dados ); 
			/*
				if pessoa.data.is_some() {
					println!("Pessoa:{} - nome:{} : dados:{:?} ", pessoa.id, pessoa.name, pessoa.data ); 
				} else {
					println!("Pessoa:{} - nome:{} : dados:{} ", pessoa.id, pessoa.name, "" );  
				}
			*/
		}
		
		println!(" ======================   " );

		let my_vec: Vec<u8> = vec![72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]; // "Hello World" in ASCII
		let vec_to_string = String::from_utf8(my_vec).unwrap(); // Converting to string
		println!("{}", vec_to_string); // Output: Hello World
		println!(" ======================   " );

		Ok(())

	}
}