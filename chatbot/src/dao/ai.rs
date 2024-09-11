pub mod chatbot {
	use rusqlite::{Connection, Result};

	struct Resposta {
		fid: i64,
		rid: i64,
		pergunta: String,
		relevancia: i64,
		tamanho_fid: i64,
		pergunta_correta: String,
		probab: f64,
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
		fn tamanho_fid(&self) -> &i64 {
			&self.tamanho_fid
		}
		fn pergunta_correta(&self) -> &String {
			&self.pergunta_correta
		}
		fn probab(&self) -> &f64 {
			&self.probab
		}
		fn ordem(&self) -> &i64 {
			&self.ordem
		}
		fn resposta(&self) -> &String {
			&self.resposta
		}
	}
 
	enum Classificacao {
		Alta,
		Media,
		Baixa,
		Improvavel,
	}

	fn classifica_probabilidade(probab: f64) ->  Result<Classificacao, String> { 
		if probab >= 99.74 {
			Ok(Classificacao::Alta)
		} else if probab >= 95.44 && probab < 99.74  {
			Ok(Classificacao::Media)
		} else if probab >= 68.26 && probab < 95.44 {
			Ok(Classificacao::Baixa)
		} else if probab < 68.26 {
			Ok(Classificacao::Improvavel)
		} else {
			// Handle unexpected values here (optional)
			Err("Invalid probability value".to_string())
		}
	}

	pub fn perguntar(entrada: String)  -> Result<()> {
		// println!("Você digitou: {}", entrada);
		let palavras: Vec<&str> = entrada.split(' ').collect();
		// let   frase = " ('O',1 ) ,
		// ('que',2),
		// ('é',3),
		// ('Cota',4),
		// ('Patronal',5)";

		let mut frase = String::new();

		for (indice, palavra) in palavras.iter().enumerate() {
			let mut ind = if indice>0 { "," } else{ ""};
			frase.push_str(&format!(" {}('{}', {} ) ",ind, palavra.trim(), indice + 1 ));
		}
		// println!("{}", frase);
		//let _path = "consciencia/estimulo.db";
		let conn: Connection = Connection::open("./consciencia/estimulo.db")?;
		let   queryp1 = "WITH P1 AS (
				SELECT column1 AS palavra, column2 AS ordem
				FROM (VALUES";

		let   queryp2 =") AS palavras  
			),
			P2 AS (
					SELECT DISTINCT p2.pid, p2.fid,  p2.rid, p2.pergunta
					, ROW_NUMBER()  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS ordem 
					, COUNT(p2.fid)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_fid 
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
					, ptd.tamanho_fid
				FROM P2 ptd  
				INNER JOIN P1 P1 ON(P1.ordem = ptd.ordem AND P1.palavra = ptd.pergunta)
				GROUP BY  ptd.fid, ptd.rid, ptd.pergunta, ptd.tamanho_fid ORDER BY ptd.pid ASC, ptd.fid ASC  
			),
			P4 AS (
			SELECT DISTINCT  
				ptd.fid
				, ptd.rid
				, ptd.pergunta
				--, GROUP_CONCAT(ptd.pergunta, ' ') OVER (PARTITION BY ptd.fid) AS pergunta 
				, COUNT(ptd.fid) OVER (PARTITION BY ptd.fid ) AS relevancia 
				, ROW_NUMBER() OVER (PARTITION  BY ptd.fid, ptd.rid ORDER BY ptd.fid ASC, ptd.rid ASC) AS ordem 
				FROM P3 ptd 
				ORDER BY ptd.fid DESC, ptd.rid ASC
			) -- SELECT * FROM P4 ptd ORDER BY ptd.fid DESC, ptd.rid ASC 
			,
			P5 AS (
			SELECT DISTINCT  
				ptd.fid
				, ptd.rid
				, GROUP_CONCAT(ptd.pergunta, ' ') OVER (PARTITION BY ptd.fid) AS pergunta 
				, ptd.relevancia    
				, ptd.ordem 
				FROM P4 ptd GROUP BY  ptd.fid, ptd.rid, ptd.pergunta, ptd.relevancia,  ptd.ordem
				ORDER BY ptd.relevancia DESC, ptd.ordem ASC
			),
			pergunta_encontrada AS ( 
				SELECT p.pid, p.fid, p.rid  
				, COUNT(p.pid) OVER (PARTITION BY  p.fid, p.rid ORDER BY p.fid ASC, p.rid ASC) 	AS tamanho_fid 
				, GROUP_CONCAT(p.pergunta, ' ') OVER (PARTITION BY  p.fid, p.rid ORDER BY p.fid ASC, p.rid ASC) 	AS pergunta_correta 
				FROM pergunta p  
			)
			SELECT DISTINCT ptd.fid, ptd.rid, ptd.pergunta, ptd.relevancia, pe.tamanho_fid, pe.pergunta_correta, CAST((((ptd.relevancia*100.0)/pe.tamanho_fid) ) AS DECIMAL(2,11) ) AS probab, ptd.ordem  , r.resposta  
			FROM P5 ptd
			INNER JOIN resposta r ON(ptd.rid = r.rid)
			INNER JOIN pergunta_encontrada pe ON(ptd.fid = pe.fid )
			GROUP BY ptd.fid, ptd.rid,  ptd.pergunta  ORDER BY ptd.relevancia DESC LIMIT 1;";
		let query = format!("{} {} {}", queryp1, frase, queryp2);

		// let mut query = "WITH P1 AS (
		// 		SELECT column1 AS palavra, column2 AS ordem
		// 		FROM (VALUES
		// 			('O',1 ) ,
		// 			('que',2),
		// 			('é',3),
		// 			('Cota',4),
		// 			('Patronal',5)
		// 		";			
		let mut stmt = conn.prepare( query.as_str() , )?;
		let resps = stmt.query_map([], |row| {
			Ok(Resposta {
				fid: row.get(0)?,
				rid: row.get(1)?,
				pergunta: row.get(2)?,
				relevancia: row.get(3)?,
				tamanho_fid: row.get(4)?,
				pergunta_correta: row.get(5)?,
				probab: row.get(6)?,
				ordem: row.get(7)?,
				resposta: row.get(8)?, 
			})
		})?;
		for resp in resps.into_iter()  {
			let resposta = resp.unwrap();
			
			let probab = resposta.probab().clone();
 
			match classifica_probabilidade(probab) {
				Ok(Classificacao::Alta) => println!("R: {:?}",	 resposta.resposta()	),
				Ok(Classificacao::Media) => println!("P: Provavelmente você quis dizer {:?} \n\rR: {:?}  ", resposta.pergunta_correta(), resposta.resposta()	),
				Ok(Classificacao::Baixa) => println!("P: Acho que você quis dizer {:?} \n\rR: {:?}  ",	resposta.pergunta_correta(), resposta.resposta()	),
				Ok(Classificacao::Improvavel) => println!("Não consegui entender o que você quis dizer."),
				Err(erro) => println!("Ocorreu um erro: {}", erro),
			}
		}
		Ok(())
	}

}