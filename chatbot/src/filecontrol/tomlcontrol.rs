pub mod tomlctrl {
    use encoding::all::ASCII;
    use encoding::types::RawEncoder;
    use encoding::{ByteWriter, DecoderTrap, EncoderTrap, Encoding};
    use encoding_rs_io::DecodeReaderBytesBuilder;
    use std::fs;
    use std::fs::File;
    use std::io;
    use std::io::prelude::*;
    use std::io::{BufRead, BufReader, Write};
    use std::iter::Iterator;
    use std::env;
    use std::error::Error;
    use std::io::Read;
    extern crate encoding_rs;
    use encoding_rs::*;
    extern crate encoding_rs_io;
    use encoding_rs_io::DecodeReaderBytes;

    use toml::Value;

    use rusqlite::{Connection, Result};

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

	pub fn get_respostas(rtext: String) -> Result<Vec<Resposta>>  {
        let conn = Connection::open("./consciencia/estimulo.db")?;
        let mut stmt = conn.prepare("SELECT rid, rtext FROM resposta WHERE rtext = ?1")?;
        let respos = stmt.query_map(&[&rtext], |row| {
            Ok(Resposta {
                rid: row.get(0)?,
                rtext: row.get(1)?,
            })
        })?;
        let mut respostas = Vec::new();
        for respo in respos {
            let respo = respo?;
            println!("rid: {}, rtext: {}", respo.rid, respo.rtext);
            respostas.push(respo);
        }
        println!("Tamanho: {}", respostas.len());
        Ok(respostas)
	}

    pub fn carregarDadosTOML() -> Result<(), Box<dyn std::error::Error>> {
        let conn: Connection = Connection::open("./consciencia/estimulo.db")?;
        let file_path = "./dados/dados.toml";
        println!("In file {}", file_path);

        match fs::read_to_string(file_path) {
            Ok(contents) => {
                let toml_value: Value = toml::from_str(&contents)?;

                let dados = toml_value["dados"]["resposta"].as_array().unwrap();

                for item in dados {
                    // Begin : registra uma resposta
                    let rid = item.get("rid").and_then(|v| v.as_integer()).unwrap_or(0);
                    let rtext = item.get("rtext").and_then(|v| v.as_str()).unwrap_or("Missing");
                    let rpt : Resposta = Resposta {
                        rid: 0,
                        rtext: rtext.trim().to_string(),
                    };
                    let mut _continue_ = true;
                    match get_respostas(rtext.trim().to_string()) {
                        Ok(lista) => {
                            if(lista.len() > 0){
                            }else{
                                _continue_ = false;
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                        }
                    }
                    if(_continue_){
                        conn.execute( "INSERT INTO resposta (rtext) VALUES (?1)", (&rpt.rtext,), )?;
                        let last_id: i64 = conn.last_insert_rowid();
                        println!("last_id : {}", last_id.to_string());
                        // End : =======================================================================

                        let mut cnt: i64 = 0;
                        // println!("rid: {}", rid);
                        // println!("rtext: {}", rtext);
                        let perguntas = item.get("pergunta").and_then(|v| v.as_array()).unwrap();
                        for pergunta in perguntas {
                            cnt +=1;
                            //let rid = pergunta.get("rid").and_then(|v| v.as_integer()).unwrap_or(0);
                            let _rid = pergunta.get("rid").and_then(|v| v.as_integer()).unwrap_or(0);
                            let fid = pergunta.get("fid").and_then(|v| v.as_integer()).unwrap_or(0);
                            let ptext = pergunta.get("ptext").and_then(|v| v.as_str()).unwrap_or("Missing");

                            let perg = ptext.clone();
                            let resps: Vec<&str> = perg.split(' ').collect();
                            for (indice, palavra) in resps.iter().enumerate() {
                                let pgt : Pergunta = Pergunta {
                                    pid: 0,
                                    rid: last_id,
                                    fid: cnt,
                                    ptext: palavra.trim().to_string(),
                                };
                                conn.execute( "INSERT INTO pergunta (rid, fid, ptext) values (?1, ?2, ?3)", (&pgt.rid,&pgt.fid,&pgt.ptext,), )?;
                            }
                            // println!("rid: {}", _rid);
                            // println!("fid: {}", fid);
                            // println!("rtext: {}", ptext);
                        }
                    }

                }
            }
            Err(e) => {
                eprintln!("Erro ao ler o arquivo: {}", e);
            }
        }

        Ok(())
    }

}