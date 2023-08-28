pub mod migrationtable {
	use rusqlite::{Connection, Result};

	pub fn migration_create_table()  -> Result<()> {
		let conn = Connection::open("my_db.db")?;
		let _r1 = conn.execute( "CREATE TABLE IF NOT EXISTS foo(x INTEGER);",[], );
		let _r2 = conn.execute( "CREATE TABLE IF NOT EXISTS bar(y TEXT);",[], );
		let _r3 = conn.execute( "CREATE TABLE IF NOT EXISTS cat_colors ( id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE );",[], )?;
		let _r4 = conn.execute( "CREATE TABLE IF NOT EXISTS cats ( id INTEGER PRIMARY KEY, name TEXT NOT NULL, color_id integer NOT NULL REFERENCES cat_colors(id) );",[], )?;
		let _r5 = conn.execute( "CREATE TABLE IF NOT EXISTS person ( id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, data BLOB ); ",[], )?;
		Ok(())
	}
}