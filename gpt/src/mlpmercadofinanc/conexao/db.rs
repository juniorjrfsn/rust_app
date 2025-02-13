use rusqlite::{params, Connection, Result as SqliteResult};
use bincode;
use thiserror::Error;

// Definição do tipo de erro personalizado para o banco de dados
#[derive(Error, Debug)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error), // Mapeia erros do SQLite
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error), // Mapeia erros do Bincode
    #[error("Other database error: {0}")] // Erro genérico adicional
    Other(String),
}

// Inicializa o banco de dados e cria as tabelas necessárias
pub fn init_db() -> Result<Connection, DbError> {
    let conn = Connection::open("dados/previsoes.db").map_err(DbError::SqliteError)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS modelos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo_financeiro TEXT NOT NULL,
            modelo BLOB NOT NULL,
            means BLOB NOT NULL,
            stds BLOB NOT NULL,
            label_mean REAL NOT NULL,
            label_std REAL NOT NULL
        )",
        [],
    )
    .map_err(DbError::SqliteError)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo_financeiro TEXT NOT NULL,
            data_prevista TEXT NOT NULL,
            valor_previsto REAL NOT NULL
        )",
        [],
    )
    .map_err(DbError::SqliteError)?;

    Ok(conn)
}

// Insere um modelo no banco de dados
pub fn insert_modelo(
    conn: &Connection,
    ativo_financeiro: &str,
    modelo_serializado: &[u8],
    means: &[f64],
    stds: &[f64],
    label_mean: f64,
    label_std: f64,
) -> Result<(), DbError> {
    let means_serialized = bincode::serialize(means).map_err(DbError::BincodeError)?;
    let stds_serialized = bincode::serialize(stds).map_err(DbError::BincodeError)?;

    conn.execute(
        "INSERT INTO modelos (ativo_financeiro, modelo, means, stds, label_mean, label_std)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            ativo_financeiro,
            modelo_serializado,
            means_serialized,
            stds_serialized,
            label_mean,
            label_std
        ],
    )
    .map_err(DbError::SqliteError)?;

    Ok(())
}

// Recupera um modelo do banco de dados
pub fn get_modelo(
    conn: &Connection,
    ativo_financeiro: &str,
) -> Result<Option<(Vec<u8>, Vec<f64>, Vec<f64>, f64, f64)>, DbError> {
    let mut stmt = conn
        .prepare(
            "SELECT modelo, means, stds, label_mean, label_std FROM modelos WHERE ativo_financeiro = ?1",
        )
        .map_err(DbError::SqliteError)?;

    let mut rows = stmt.query_map(params![ativo_financeiro], |row| {
        // Recupera os dados serializados
        let means_serialized: Vec<u8> = row.get(1).map_err(DbError::SqliteError)?;
        let stds_serialized: Vec<u8> = row.get(2).map_err(DbError::SqliteError)?;

        // Desserializa os dados
        let means: Vec<f64> = bincode::deserialize(&means_serialized).map_err(DbError::BincodeError)?;
        let stds: Vec<f64> = bincode::deserialize(&stds_serialized).map_err(DbError::BincodeError)?;

        Ok((
            row.get(0).map_err(DbError::SqliteError)?, // modelo
            means,
            stds,
            row.get(3).map_err(DbError::SqliteError)?, // label_mean
            row.get(4).map_err(DbError::SqliteError)?, // label_std
        ))
    })
    .map_err(DbError::SqliteError)?;

    if let Some(row) = rows.next() {
        Ok(Some(row.map_err(DbError::SqliteError)?))
    } else {
        Ok(None)
    }
}

// Insere uma previsão no banco de dados
pub fn insert_previsao(
    conn: &Connection,
    ativo_financeiro: &str,
    data_prevista: &str,
    valor_previsto: f64,
) -> Result<(), DbError> {
    conn.execute(
        "INSERT INTO previsoes (ativo_financeiro, data_prevista, valor_previsto)
         VALUES (?1, ?2, ?3)",
        params![ativo_financeiro, data_prevista, valor_previsto],
    )
    .map_err(DbError::SqliteError)?;

    Ok(())
}