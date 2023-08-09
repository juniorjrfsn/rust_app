pub mod migrationtable {
	use rusqlite::{Connection, Result};

	pub fn migration_create_table()  -> Result<()> {
		let conn = Connection::open("my_db.db")?;
		let _r1 = conn.execute( "CREATE TABLE IF NOT EXISTS foo(x INTEGER);",[], );
		let _r2 = conn.execute( "CREATE TABLE IF NOT EXISTS bar(y TEXT);",[], );
		let _r3 = conn.execute( "CREATE TABLE IF NOT EXISTS cat_colors ( id integer primary key, name text not null unique );",[], )?;
		let _r4 = conn.execute( "CREATE TABLE IF NOT EXISTS cats ( id integer primary key, name text not null, color_id integer not null references cat_colors(id) );",[], )?; 
		let _r5 = conn.execute( "CREATE TABLE person ( id INTEGER PRIMARY KEY, name TEXT NOT NULL, data BLOB ); ",[], )?; 
		Ok(())
	}

}