use std::io;
use std::io::{BufRead, BufReader, Write};
use std::fs::File;
use std::io::prelude::*;
use encoding_rs_io::DecodeReaderBytesBuilder;
use encoding::{Encoding, ByteWriter, EncoderTrap, DecoderTrap};
use encoding::types::RawEncoder;
use encoding::all::ASCII;
use std::fs;

use std::iter::Iterator;

use std::env;
use std::error::Error;
use std::io::Read;

extern crate encoding_rs;
use encoding_rs::*;
extern crate encoding_rs_io;
use encoding_rs_io::DecodeReaderBytes;

use toml::Value;

fn main() -> Result<(), std::io::Error> {



    let file_path = "arquivosplit.toml";
    // --snip--
    println!("In file {}", file_path);
 
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    
    let       toml_value: Value = toml::from_str(contents.as_str()).unwrap();
    let arquivos = toml_value["dados"]["arquivo"].as_array().unwrap();
    
    for arquivo in arquivos {
        let nome_origem = arquivo.get("arquivo_nomeorigem").and_then(|v| v.as_str()).unwrap_or("Missing");
        let qtde_parte = arquivo.get("arquivo_qtde_parte").and_then(|v| v.as_integer()).unwrap_or(0);
        let nome_parte = arquivo.get("arquivo_nome_parte").and_then(|v| v.as_str()).unwrap_or("Missing");

        println!("Nome do arquivo de origem: {}", nome_origem);
        println!("Quantidade de partes: {}", qtde_parte);
        println!("Nome da parte: {}", nome_parte);
        println!("----------------------"); // Separator between entries
    }
   
    let mut tabela: toml::Value = toml::from_str(contents.as_str()).unwrap();
    let arquivo_nomeorigem = tabela["dados"]["arquivo"][0]["arquivo_nomeorigem"].as_str().unwrap();
    let arquivo_qtde_parte = tabela["dados"]["arquivo"][0]["arquivo_qtde_parte"].as_integer().unwrap();
    let arquivo_nome_parte = tabela["dados"]["arquivo"][0]["arquivo_nome_parte"].as_str().unwrap();

    let file = File::open(arquivo_nomeorigem).expect("Não foi possível abrir o arquivo");

    let reader = BufReader::new(file);
    let decoder = WINDOWS_1252.new_decoder();

    let mut decode_reader = DecodeReaderBytesBuilder::new()
    .encoding(Some(&WINDOWS_1252))  // Specify the encoding here
    .build(reader);

    let mut texto_utf8 = String::new();
    decode_reader.read_to_string(&mut texto_utf8).unwrap();

    let mut numero_linhas = 0;
    for linha in texto_utf8.lines() {
        numero_linhas += 1;
        match linha {
            str => {
                println!("{}", linha);
            } 
        } 
    }
    println!("linhas : {}", numero_linhas);

    let num_lin_por_arq = (numero_linhas / arquivo_qtde_parte);
    println!("Qtde de linhas por arquivo: {}", num_lin_por_arq);
    Ok(())
}
 