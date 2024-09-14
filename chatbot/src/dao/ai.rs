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
		let   queryp1 = "WITH Entrada AS (
			SELECT column1 AS ptext, column2 AS ordem
			FROM (VALUES";

		let   queryp2 ="    ) AS ptexts
			),
					EntradaOrdenada AS (
				SELECT DISTINCT p2.rid, p2.fid, p2.pid, P1.ordem, P1.ptext
					, ROW_NUMBER()  OVER (PARTITION  BY  p2.rid, p2.fid ORDER BY  p2.rid ASC, p2.fid ASC, P1.ordem ASC ) AS entradaordem
					, COUNT(p2.fid)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_fid
					, COUNT(P1.ordem)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_entrada

				FROM Entrada P1
				INNER JOIN pergunta p2 ON( P1.ptext = p2.ptext)
				GROUP BY p2.rid, p2.fid, p2.pid, P1.ordem, P1.ptext ORDER BY p2.rid ASC, p2.fid ASC, P1.ordem ASC,  p2.pid ASC
			)
			,
			PerguntaOrdenada AS (
				SELECT DISTINCT p2.rid, p2.fid, p2.pid,  p2.ptext
				,  ROW_NUMBER()  OVER (PARTITION  BY  p2.rid, p2.fid ORDER BY  p2.rid ASC, p2.fid ASC) AS perguntaordem
				FROM EntradaOrdenada eo
				INNER JOIN pergunta p2 ON(eo.rid = p2.rid AND eo.fid = p2.fid AND eo.ptext = p2.ptext) ORDER BY p2.rid ASC
			),
			Filtro AS (
				SELECT  po.*
				FROM PerguntaOrdenada po
				INNER JOIN EntradaOrdenada eo ON(po.rid = eo.rid AND po.fid = eo.fid AND po.ptext = eo.ptext AND po.perguntaordem = eo.entradaordem)
				ORDER BY po.rid ASC
			),
			Relevanciaresposta AS (
				SELECT DISTINCT re.rid, re.fid, re.ptext, COUNT(re.rid) OVER(PARTITION BY re.rid, re.fid ) AS relevancia_frase
				FROM Filtro re
				ORDER BY re.rid ASC
			),
			Relevancia AS (
				SELECT DISTINCT re.rid, re.fid, re.ptext
				, COUNT(re.rid) OVER(PARTITION BY re.rid ) AS relevancia_resposta
				FROM Relevanciaresposta re
				GROUP BY re.rid, re.ptext
				ORDER BY re.rid ASC
			),
			probabilidade AS (
				SELECT fi.rid,fi.fid
				, MAX(rele.relevancia_resposta) AS relevancia_resposta
				, COUNT(fi.fid) OVER(PARTITION BY fi.rid ) AS qtd_frase
				, MAX(rr.relevancia_frase) AS relevancia_frase
				--, GROUP_CONCAT(p2.ptext, ' ') OVER (PARTITION BY  p2.fid ORDER BY p2.fid ASC, p2.pid ASC) 	AS pergunta_correta
				FROM Relevancia rele
				INNER JOIN Filtro fi ON(rele.rid = fi.rid)
				INNER JOIN pergunta p2 ON(fi.fid = p2.fid)
				INNER JOIN Relevanciaresposta rr ON(rele.rid = rr.rid)
				INNER JOIN EntradaOrdenada eo ON(p2.rid = eo.rid AND p2.fid = eo.fid )
				GROUP BY fi.rid,fi.fid
			),
			perguntando AS (
				SELECT co.relevancia_resposta, co.rid,co.fid, co.qtd_frase, CAST((((co.relevancia_resposta*100.0)/7) ) AS DECIMAL(2,11) ) AS probab, co.relevancia_frase
				FROM probabilidade co
				ORDER BY co.relevancia_resposta DESC
			),
			pesando AS (
				SELECT DISTINCT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, GROUP_CONCAT(perg.ptext, ' ') OVER (PARTITION BY co.rid,co.fid) AS ptext 
				FROM perguntando co
				INNER JOIN pergunta perg ON(co.rid = perg.rid AND co.fid = perg.fid)
				WHERE co.probab >= 68.26
				ORDER BY co.relevancia_resposta DESC
			),
			Conclusao AS (

			SELECT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, co.ptext, re.rtext 
			FROM pesando co
			INNER JOIN resposta re ON(co.rid = re.rid )
			)
			SELECT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, co.ptext, co.rtext FROM Conclusao coLIMIT 1;";
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
				rid: row.get(0)?,
				fid: row.get(1)?,
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