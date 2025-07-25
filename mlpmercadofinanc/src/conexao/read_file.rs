// file : src/conexao/read_file.rs

use csv::ReaderBuilder;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::fs;
use thiserror::Error;

/// Custom error types for data handling.
#[derive(Error, Debug)]
pub enum DataError {
    /// Represents an I/O error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Represents a CSV parsing error.
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    /// Represents an invalid data format error.
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
}

/// Struct to deserialize data from "investing" source.
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

/// Struct to deserialize data from "infomoney" source.
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

/// Trait to provide a consistent way to convert deserialized records into a `Vec<String>`.
trait CsvRecord {
    fn into_row(self) -> Vec<String>;
}

/// Implementation of `CsvRecord` for `ObjectInvestingHistCotacaoAtivo`.
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

/// Implementation of `CsvRecord` for `Record`.
impl CsvRecord for Record {
    fn into_row(self) -> Vec<String> {
        // Ensure consistent column order
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

/// Generic function to read a CSV file and deserialize its records.
///
/// # Arguments
///
/// * `file_path` - The path to the CSV file.
///
/// # Returns
///
/// A `Result` containing a `Vec<Vec<String>>` representing the CSV data, or a `DataError`.
fn read_csv_generic<T: DeserializeOwned + CsvRecord>(
    file_path: &str,
) -> Result<Vec<Vec<String>>, DataError> {
    // Check if the file exists
    if !fs::metadata(file_path).is_ok() {
        return Err(DataError::InvalidDataFormat(format!(
            "Arquivo não encontrado: {}",
            file_path
        )));
    }
    let file = fs::File::open(file_path).map_err(DataError::IoError)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut matrix: Vec<Vec<String>> = Vec::new();

    // Deserialize each record and convert it into a consistent row format
    for result in rdr.deserialize() {
        let record: T = result.map_err(DataError::CsvError)?;
        matrix.push(record.into_row());
    }
    Ok(matrix)
}

/// Public function to read CSV data based on the specified source.
///
/// # Arguments
///
/// * `file_path` - The path to the CSV file.
/// * `cotac_fonte` - The data source (e.g., "investing", "infomoney").
/// * `_ativo_financeiro` - Financial asset ticker (currently unused).
///
/// # Returns
///
/// A `Result` containing a `Vec<Vec<String>>` representing the CSV data, or a `DataError`.
pub fn ler_csv(
    file_path: &str,
    cotac_fonte: &str,
    _ativo_financeiro: &str,
) -> Result<Vec<Vec<String>>, DataError> {
    let matrix = match cotac_fonte {
        "investing" => read_csv_generic::<ObjectInvestingHistCotacaoAtivo>(file_path)?,
        "infomoney" => read_csv_generic::<Record>(file_path)?,
        _ => {
            return Err(DataError::InvalidDataFormat(format!(
                "Fonte de cotação desconhecida: {}",
                cotac_fonte
            )))
        }
    };
    // Validate the matrix: ensure it's not empty and all rows have 7 columns
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != 7) {
        return Err(DataError::InvalidDataFormat(
            "O arquivo CSV não contém dados suficientes ou formato inválido.".to_string(),
        ));
    }
    Ok(matrix)
}
