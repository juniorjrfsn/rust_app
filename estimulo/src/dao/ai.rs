pub mod chatbot {
	use rusqlite::{Connection, Result};

	struct Resposta {
		fid: i64,
		rid: i64,
		pergunta: String,
		relevancia: i64,
		ordem: i64,
		resposta: String, 
	} 
	impl Resposta {
		fn fid(&self) -> &i64 {
			&self.fid
		}
		fn rid(&self) -> &i64 {
			&self.rid
		}
		fn pergunta(&self) -> &String {
			&self.pergunta
		}
		fn relevancia(&self) -> &i64 {
			&self.relevancia
		}
		fn ordem(&self) -> &i64 {
			&self.ordem
		}
		fn resposta(&self) -> &String {
			&self.resposta
		}
	}

	pub fn perguntar()  -> Result<()> {
		//let _path = "consciencia/estimulo.db";
		let conn: Connection = Connection::open("./consciencia/estimulo.db")?; 
		let mut stmt = conn.prepare( "WITH P1 AS (
    SELECT column1 AS palavra, column2 AS ordem
    FROM (VALUES
        ('O',1 ) ,
        ('que',2),
        ('é',3),
        ('Cota',4),
        ('Patronal',5)
    ) AS palavras  
),
P2 AS (
        SELECT DISTINCT p2.pid, p2.fid,  p2.rid, p2.pergunta, ROW_NUMBER()  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS ordem 
    FROM P1 P1 
    INNER JOIN pergunta p2 ON( P1.palavra = p2.pergunta) 
    GROUP BY p2.pid, p2.fid, p2.rid, p2.pergunta ORDER BY p2.pid ASC, p2.fid ASC  
) 
,
P3 AS (
    SELECT DISTINCT  
        ptd.fid
        , ptd.rid
        , ptd.pergunta   
    FROM P2 ptd  
    INNER JOIN P1 P1 ON(P1.ordem = ptd.ordem AND P1.palavra = ptd.pergunta)
    GROUP BY  ptd.fid, ptd.rid, ptd.pergunta ORDER BY ptd.pid ASC, ptd.fid ASC  
),
P4 AS (
   SELECT DISTINCT  
     ptd.fid
    , ptd.rid
    , ptd.pergunta
    , COUNT(ptd.fid)    OVER (PARTITION BY ptd.fid ) AS relevancia  
    , ROW_NUMBER()      OVER (PARTITION  BY ptd.fid, ptd.rid ORDER BY ptd.fid ASC, ptd.rid ASC) AS ordem 
    FROM P3 ptd
),
P5 AS (
   SELECT DISTINCT  
     ptd.fid
    , ptd.rid
    , GROUP_CONCAT(ptd.pergunta, ' ') OVER (PARTITION BY ptd.fid ) AS pergunta 
    , ptd.relevancia  
    , ptd.ordem 
    FROM P4 ptd  ORDER BY ptd.relevancia DESC, ptd.ordem ASC
)
SELECT ptd.fid, ptd.rid, ptd.pergunta, ptd.relevancia, ptd.ordem, r.resposta  
FROM P5 ptd
INNER JOIN resposta r ON(ptd.rid = r.rid)
GROUP BY ptd.fid, ptd.rid, ptd.pergunta  ORDER BY ptd.relevancia DESC;", )?;
		let resps = stmt.query_map([], |row| {
			Ok(Resposta {
				fid: row.get(0)?,
				rid: row.get(1)?,
				pergunta: row.get(2)?,
				relevancia: row.get(3)?,
				ordem: row.get(4)?,
				resposta: row.get(5)?, 
			})
		})?;
		for resp in resps.into_iter()  {
			let resposta = resp.unwrap();
			println!("p: {:?} r: {:?}  ", resposta.pergunta(), resposta.resposta() );
		}
		Ok(())
	}
}