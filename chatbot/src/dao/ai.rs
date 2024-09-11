pub mod chatbot {
	use std::str;
	use std::error::Error;
	use std::fs;
	use std::fs::{File, OpenOptions};
	use std::io;
	use std::io::prelude::*;
	#[cfg(target_family = "unix")]
	use std::os::unix;
	#[cfg(target_family = "windows")]
	use std::os::windows;
	use std::path::Path;

	use rusqlite::{Connection, Result};

	struct Respostabot {
		fid: i64,
		rid: i64,
		ptext: String,
		relevancia: i64,
		tamanho_fid: i64,
		pergunta_correta: String,
		probab: f64,
		ordem: i64,
		rtext: String,
	}



	impl Respostabot {
		fn fid(&self) -> &i64 {
			&self.fid
		}
		fn rid(&self) -> &i64 {
			&self.rid
		}
		fn ptext(&self) -> &String {
			&self.ptext
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
		fn rtext(&self) -> &String {
			&self.rtext
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



	#[derive(Debug)]
	struct Resposta {
		rid	: i64,
		rtext: String,
	}

	#[derive(Debug)]
	struct Pergunta {
		pid	: i64,
		rid	: i64,
		fid	: i64,
		ptext: String,
	}
	pub fn treinar(resp: String)  -> Result<()> {
		// let resps: Vec<&str> = resp.split(' ').collect();
		// let mut respostta = String::new();
		// for (indice, palavra) in resps.iter().enumerate() {
		// 	let mut ind = if indice>0 { "," } else{ ""};
		// 	respostta.push_str(&format!(" {}('{}', {} ) ",ind, palavra.trim(), indice + 1 ));
		// }
		// println!("{}", respostta);
		let conn: Connection = Connection::open("./consciencia/estimulo.db")?;
		let rpt : Resposta = Resposta {
			rid: 0,
			rtext: resp.trim().to_string(),
		};
		conn.execute( "INSERT INTO resposta (resposta) VALUES (?1)", (&rpt.rtext,), )?;
		let last_id: i64 = conn.last_insert_rowid();
		println!("last_id : {}", last_id.to_string());

		let mut cnt: i64 = 0;
		loop{
			cnt +=1;

			let mut escolha = String::new();
			let mut character: char = 'N';
			match io::stdin().read_line(&mut escolha) {
				Ok(_) => {
					character = match escolha.trim().chars().next() {
						Some(c) if c.is_alphabetic() && (c == 'S' || c == 's' || c == '1') => c,
						_ => {
							println!("Caractere inválido. Considerando 'S' como padrão.");
							'N'
						}
					};
				}
				Err(error) => {
					println!("Ocorreu um erro durante a leitura: {}", error);
				}
			}
			let var_name = if character == 'S' || character == 's' || character == '1'   {
				let mut per = String::new();
                io::stdin().read_line(&mut per).expect("Falha ao ler a linha");
				let perg = per.clone();

				let resps: Vec<&str> = perg.split(' ').collect();
				for (indice, palavra) in resps.iter().enumerate() {
					let pgt : Pergunta = Pergunta {
						pid: 0,
						rid: last_id,
						ptext: palavra.trim().to_string(),
						fid: cnt,
					};
					conn.execute( "INSERT INTO pergunta (rid, fid, ptext) values (?1, ?2, ?3)", (&pgt.rid,&pgt.fid,&pgt.ptext,), )?;
					//respostta.push_str(&format!(" {}('{}', {} ) ",ind, palavra.trim(), indice + 1 ));
				}

			}else{
				break;
			};

			println!("Digite S ou 1 para escrever uma pergunta:");
		};
		Ok(())
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
			let ind = if indice>0 { "," } else{ ""};
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
					SELECT DISTINCT p2.pid, p2.fid,  p2.rid, p2.ptext
					, ROW_NUMBER()  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS ordem
					, COUNT(p2.fid)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_fid
				FROM P1 P1
				INNER JOIN pergunta p2 ON( P1.palavra = p2.ptext)
				GROUP BY p2.pid, p2.fid, p2.rid, p2.ptext ORDER BY p2.pid ASC, p2.fid ASC
			)
			,
			P3 AS (
				SELECT DISTINCT
					ptd.fid
					, ptd.rid
					, ptd.ptext
					, ptd.tamanho_fid
				FROM P2 ptd
				INNER JOIN P1 P1 ON(P1.ordem = ptd.ordem AND P1.palavra = ptd.ptext)
				GROUP BY  ptd.fid, ptd.rid, ptd.ptext, ptd.tamanho_fid ORDER BY ptd.pid ASC, ptd.fid ASC
			),
			P4 AS (
			SELECT DISTINCT
				ptd.fid
				, ptd.rid
				, ptd.ptext
				--, GROUP_CONCAT(ptd.ptext, ' ') OVER (PARTITION BY ptd.fid) AS ptext
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
				, GROUP_CONCAT(ptd.ptext, ' ') OVER (PARTITION BY ptd.fid) AS ptext
				, ptd.relevancia
				, ptd.ordem
				FROM P4 ptd GROUP BY  ptd.fid, ptd.rid, ptd.ptext, ptd.relevancia,  ptd.ordem
				ORDER BY ptd.relevancia DESC, ptd.ordem ASC
			),
			pergunta_encontrada AS (
				SELECT p.pid, p.fid, p.rid
				, COUNT(p.pid) OVER (PARTITION BY  p.fid, p.rid ORDER BY p.fid ASC, p.rid ASC) 	AS tamanho_fid
				, GROUP_CONCAT(p.ptext, ' ') OVER (PARTITION BY  p.fid, p.rid ORDER BY p.fid ASC, p.rid ASC) 	AS pergunta_correta
				FROM pergunta p
			)
			SELECT DISTINCT ptd.fid, ptd.rid, ptd.ptext, ptd.relevancia, pe.tamanho_fid, pe.pergunta_correta, CAST((((ptd.relevancia*100.0)/pe.tamanho_fid) ) AS DECIMAL(2,11) ) AS probab, ptd.ordem  , r.rtext
			FROM P5 ptd
			INNER JOIN resposta r ON(ptd.rid = r.rid)
			INNER JOIN pergunta_encontrada pe ON(ptd.fid = pe.fid )
			GROUP BY ptd.fid, ptd.rid,  ptd.ptext  ORDER BY ptd.relevancia DESC LIMIT 1;";
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
			Ok(Respostabot {
				fid: row.get(0)?,
				rid: row.get(1)?,
				ptext: row.get(2)?,
				relevancia: row.get(3)?,
				tamanho_fid: row.get(4)?,
				pergunta_correta: row.get(5)?,
				probab: row.get(6)?,
				ordem: row.get(7)?,
				rtext: row.get(8)?,
			})
		})?;
		for resp in resps.into_iter()  {
			let resposta = resp.unwrap();

			let probab = resposta.probab().clone();

			match classifica_probabilidade(probab) {
				Ok(Classificacao::Alta) => println!("R: {:?}",	 resposta.rtext()	),
				Ok(Classificacao::Media) => println!("P: Provavelmente você quis dizer {:?} \n\rR: {:?}  ", resposta.pergunta_correta(), resposta.rtext()	),
				Ok(Classificacao::Baixa) => println!("P: Acho que você quis dizer {:?} \n\rR: {:?}  ",	resposta.pergunta_correta(), resposta.rtext()	),
				Ok(Classificacao::Improvavel) => println!("Não consegui entender o que você quis dizer."),
				Err(erro) => println!("Ocorreu um erro: {}", erro),
			}
		}
		Ok(())
	}

}