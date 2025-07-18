// file : src/conexao/read_file.rs


use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
}

#[derive(Debug, Deserialize)]
struct ObjectInvestingHistCotacaoAtivo {
    #[serde(rename = "Data")]
    data: String,
    #[serde(rename = "Último")]
    fechamento: String,
    #[serde(rename = "Abertura")]
    abertura: String,
    #[serde(rename = "Máxima")]
    maximo: String,
    #[serde(rename = "Mínima")]
    minimo: String,
    #[serde(rename = "Vol.")]
    volume: String,
    #[serde(rename = "Var%")]
    variacao: String,
}

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "DATA")]
    data: String,
    #[serde(rename = "ABERTURA")]
    abertura: String,
    #[serde(rename = "FECHAMENTO")]
    fechamento: String,
    #[serde(rename = "VARIAÇÃO")]
    variacao: String,
    #[serde(rename = "MÍNIMO")]
    minimo: String,
    #[serde(rename = "MÁXIMO")]
    maximo: String,
    #[serde(rename = "VOLUME")]
    volume: String,
}

trait CsvRecord {
    fn into_row(self) -> Vec<String>;
}

impl CsvRecord for ObjectInvestingHistCotacaoAtivo {
    fn into_row(self) -> Vec<String> {
        vec![
            self.data,
            self.abertura,
            self.fechamento,
            self.variacao,
            self.minimo,
            self.maximo,
            self.volume,
        ]
    }
}

impl CsvRecord for Record {
    fn into_row(self) -> Vec<String> {
        vec![
            self.data,
            self.abertura,
            self.fechamento,
            self.variacao,
            self.minimo,
            self.maximo,
            self.volume,
        ]
    }
}

fn read_csv_generic<T: for<'de> Deserialize<'de> + CsvRecord>(file_path: &str) -> Result<Vec<Vec<String>>, DataError> {
    if !fs::metadata(file_path).is_ok() {
        return Err(DataError::InvalidDataFormat(format!(
            "Arquivo não encontrado: {}",
            file_path
        )));
    }

    let file = fs::File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(file);

    let mut matrix = Vec::new();
    for result in rdr.deserialize() {
        let record: T = result?;
        matrix.push(record.into_row());
    }

    Ok(matrix)
}

pub fn ler_csv(
    file_path: &str,
    cotac_fonte: &str,
    _ativo_financeiro: &str,
) -> Result<Vec<Vec<String>>, DataError> {
    let matrix = match cotac_fonte {
        "investing" => read_csv_generic::<ObjectInvestingHistCotacaoAtivo>(file_path)?,
        "infomoney" => read_csv_generic::<Record>(file_path)?,
        _ => return Err(DataError::InvalidDataFormat(format!(
            "Fonte de cotação desconhecida: {}",
            cotac_fonte
        ))),
    };

    if matrix.is_empty() || matrix.iter().any(|row| row.len() != 7) {
        return Err(DataError::InvalidDataFormat(
            "O arquivo CSV não contém dados suficientes ou formato inválido.".to_string(),
        ));
    }

    Ok(matrix)
}