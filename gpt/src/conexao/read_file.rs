use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs;

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

/// Função privada para ler o CSV e transformá-lo em uma matriz de strings.
pub fn read_csv_to_matrix(file_path: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    // Verifica se o arquivo existe
    if !fs::metadata(file_path).is_ok() {
        return Err(format!("Arquivo não encontrado: {}", file_path).into());
    }

    // Abre o arquivo CSV
    let file = fs::File::open(file_path)?;

    // Cria um leitor CSV
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // Inicializa a matriz de dados
    let mut matrix: Vec<Vec<String>> = Vec::new();

    // Itera sobre os registros no arquivo CSV
    for result in rdr.deserialize() {
        // Desserializa cada linha em uma estrutura `Record`
        let record: Record = result?;

        // Converte o registro em um vetor de strings
        let row = vec![
            record.data,
            record.abertura,
            record.fechamento,
            record.variacao,
            record.minimo,
            record.maximo,
            record.volume,
        ];

        // Adiciona a linha à matriz
        matrix.push(row);
    }

    Ok(matrix)
}

/// Função pública para ler o arquivo CSV e retornar a matriz de dados.
pub fn ler_csv(file_path: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    // Lê o arquivo CSV e retorna a matriz de dados
    let matrix = read_csv_to_matrix(file_path)?;

    Ok(matrix)
}