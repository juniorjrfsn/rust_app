// file : src/conexao/read_file.rs


use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs;
use log::info;
use crate::mlp::common::AppError;

#[derive(Debug, Deserialize)]
struct FinancialRecord {
    #[serde(rename = "Data", alias = "DATA")]
    data: String,
    #[serde(rename = "Abertura", alias = "ABERTURA")]
    abertura: String,
    #[serde(rename = "Último", alias = "FECHAMENTO")]
    fechamento: String,
    #[serde(rename = "Var%", alias = "VARIAÇÃO")]
    variacao: String,
    #[serde(rename = "Mínima", alias = "MÍNIMO")]
    minimo: String,
    #[serde(rename = "Máxima", alias = "MÁXIMO")]
    maximo: String,
    #[serde(rename = "Vol.", alias = "VOLUME")]
    volume: String,
}

pub fn ler_csv(file_path: &str, cotac_fonte: &str) -> Result<Vec<Vec<String>>, AppError> {
    let file = fs::File::open(file_path).map_err(|e| AppError::IoError(e))?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(file);

    let mut matrix = Vec::new();
    for (i, result) in rdr.deserialize().enumerate() {
        let record: FinancialRecord = result.map_err(|e| AppError::CsvError(e))?;
        let row = vec![
            record.data,
            record.abertura,
            record.fechamento,
            record.variacao,
            record.minimo,
            record.maximo,
            record.volume,
        ];
        if row.len() != 7 || row.iter().any(|s| s.is_empty() || s == "n/d") {
            return Err(AppError::InvalidData {
                location: format!("row {}", i + 1),
                message: format!("Invalid row: {:?}", row),
            });
        }
        matrix.push(row);
    }

    if matrix.is_empty() {
        return Err(AppError::InvalidData {
            location: "matrix".to_string(),
            message: "No data read from file".to_string(),
        });
    }

    info!("Read {} rows from {} (source: {})", matrix.len(), file_path, cotac_fonte);
    Ok(matrix)
}