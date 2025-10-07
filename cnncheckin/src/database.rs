// file: cnncheckin/src/database.rs
// Módulo de integração com PostgreSQL

use std::error::Error;
use postgres::{Client, NoTls, Row};
use r2d2::Pool;
use r2d2_postgres::{PostgresConnectionManager, postgres::NoTls as R2D2NoTls};
use serde::{Deserialize, Serialize};

use crate::cnn_model::{TrainedModel, ModelMetadata};

type ConnectionPool = Pool<PostgresConnectionManager<R2D2NoTls>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i32,
    pub name: String,
    pub embedding: Vec<f32>,
    pub created_at: String,
    pub last_seen: Option<String>,
    pub photo_count: i32,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckIn {
    pub id: i32,
    pub person_id: i32,
    pub timestamp: String,
    pub confidence: f32,
    pub method: String,
    pub photo_path: Option<String>,
}

pub struct Database {
    pool: ConnectionPool,
}

impl Database {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let config = crate::config::Config::load()?;
        
        let manager = PostgresConnectionManager::new(
            format!(
                "host={} port={} user={} password={} dbname={}",
                config.database.host,
                config.database.port,
                config.database.username,
                config.database.password,
                config.database.database
            ).parse()?,
            R2D2NoTls,
        );

        let pool = Pool::builder()
            .max_size(config.database.max_connections)
            .build(manager)?;

        Ok(Self { pool })
    }

    pub async fn setup_tables(&self) -> Result<(), Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        // Tabela de modelos
        client.execute(
            "CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                accuracy REAL NOT NULL,
                num_classes INTEGER NOT NULL,
                num_parameters BIGINT NOT NULL,
                training_epochs INTEGER NOT NULL,
                class_names TEXT[] NOT NULL,
                model_data BYTEA NOT NULL,
                is_active BOOLEAN DEFAULT TRUE
            )",
            &[],
        )?;

        // Tabela de pessoas
        client.execute(
            "CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                embedding REAL[] NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_seen TIMESTAMP WITH TIME ZONE,
                photo_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE
            )",
            &[],
        )?;

        // Tabela de check-ins
        client.execute(
            "CREATE TABLE IF NOT EXISTS checkins (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                confidence REAL NOT NULL,
                method VARCHAR(50) NOT NULL,
                photo_path TEXT
            )",
            &[],
        )?;

        // Índices para performance
        client.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkins_person_id ON checkins(person_id)",
            &[],
        )?;
        
        client.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkins_timestamp ON checkins(timestamp)",
            &[],
        )?;

        println!("Tabelas do banco de dados configuradas com sucesso!");
        Ok(())
    }

    pub async fn save_model(&self, model: &TrainedModel) -> Result<i32, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let model_name = format!("model_{}", chrono::Utc::now().timestamp());
        
        let row = client.query_one(
            "INSERT INTO models (name, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             RETURNING id",
            &[
                &model_name,
                &model.metadata.accuracy,
                &(model.metadata.num_classes as i32),
                &(model.metadata.num_parameters as i64),
                &(model.metadata.training_epochs as i32),
                &model.metadata.class_names,
                &model.weights,
            ],
        )?;

        Ok(row.get(0))
    }

    pub async fn load_model(&self, model_id: i32) -> Result<TrainedModel, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let row = client.query_one(
            "SELECT id, name, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data
             FROM models WHERE id = $1 AND is_active = TRUE",
            &[&model_id],
        )?;

        let metadata = ModelMetadata {
            id: Some(row.get(0)),
            created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(2).to_rfc3339(),
            accuracy: row.get(3),
            num_classes: row.get::<_, i32>(4) as usize,
            num_parameters: row.get::<_, i64>(5) as usize,
            training_epochs: row.get::<_, i32>(6) as usize,
            class_names: row.get(7),
        };

        Ok(TrainedModel {
            metadata,
            weights: row.get(8),
        })
    }

    pub async fn load_latest_model(&self) -> Result<TrainedModel, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let row = client.query_one(
            "SELECT id, name, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data
             FROM models WHERE is_active = TRUE
             ORDER BY created_at DESC LIMIT 1",
            &[],
        )?;

        let metadata = ModelMetadata {
            id: Some(row.get(0)),
            created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(2).to_rfc3339(),
            accuracy: row.get(3),
            num_classes: row.get::<_, i32>(4) as usize,
            num_parameters: row.get::<_, i64>(5) as usize,
            training_epochs: row.get::<_, i32>(6) as usize,
            class_names: row.get(7),
        };

        Ok(TrainedModel {
            metadata,
            weights: row.get(8),
        })
    }

    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let rows = client.query(
            "SELECT id, name, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names
             FROM models WHERE is_active = TRUE
             ORDER BY created_at DESC",
            &[],
        )?;

        let mut models = Vec::new();
        for row in rows {
            let metadata = ModelMetadata {
                id: Some(row.get(0)),
                created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(2).to_rfc3339(),
                accuracy: row.get(3),
                num_classes: row.get::<_, i32>(4) as usize,
                num_parameters: row.get::<_, i64>(5) as usize,
                training_epochs: row.get::<_, i32>(6) as usize,
                class_names: row.get(7),
            };
            models.push(metadata);
        }

        Ok(models)
    }

    pub async fn save_person(&self, name: &str, embedding: &[f32]) -> Result<i32, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let row = client.query_one(
            "INSERT INTO persons (name, embedding)
             VALUES ($1, $2)
             ON CONFLICT (name) DO UPDATE SET 
             embedding = EXCLUDED.embedding,
             photo_count = persons.photo_count + 1
             RETURNING id",
            &[&name, &embedding],
        )?;

        Ok(row.get(0))
    }

    pub async fn find_similar_person(
        &self, 
        embedding: &[f32], 
        threshold: f32
    ) -> Result<Option<Person>, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        // Busca por similaridade de cosseno usando extensões do PostgreSQL
        // Em produção, considere usar pgvector para otimizar
        let rows = client.query(
            "SELECT id, name, embedding, created_at, last_seen, photo_count, is_active
             FROM persons WHERE is_active = TRUE",
            &[],
        )?;

        let mut best_match: Option<Person> = None;
        let mut best_similarity = 0.0f32;

        for row in rows {
            let stored_embedding: Vec<f32> = row.get(2);
            let similarity = calculate_cosine_similarity(embedding, &stored_embedding);
            
            if similarity > threshold && similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(Person {
                    id: row.get(0),
                    name: row.get(1),
                    embedding: stored_embedding,
                    created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(3).to_rfc3339(),
                    last_seen: row.get::<_, Option<chrono::DateTime<chrono::Utc>>>(4)
                        .map(|dt| dt.to_rfc3339()),
                    photo_count: row.get(5),
                    is_active: row.get(6),
                });
            }
        }

        // Atualizar last_seen se encontrou match
        if let Some(ref person) = best_match {
            client.execute(
                "UPDATE persons SET last_seen = NOW() WHERE id = $1",
                &[&person.id],
            )?;
        }

        Ok(best_match)
    }

    pub async fn record_checkin(
        &self,
        person_id: i32,
        confidence: f32,
        method: &str,
        photo_path: Option<&str>,
    ) -> Result<i32, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let row = client.query_one(
            "INSERT INTO checkins (person_id, confidence, method, photo_path)
             VALUES ($1, $2, $3, $4)
             RETURNING id",
            &[&person_id, &confidence, &method, &photo_path],
        )?;

        Ok(row.get(0))
    }

    pub async fn get_recent_checkins(&self, limit: i32) -> Result<Vec<CheckIn>, Box<dyn Error>> {
        let mut client = self.pool.get()?;
        
        let rows = client.query(
            "SELECT id, person_id, timestamp, confidence, method, photo_path
             FROM checkins
             ORDER BY timestamp DESC
             LIMIT $1",
            &[&limit],
        )?;

        let mut checkins = Vec::new();
        for row in rows {
            checkins.push(CheckIn {
                id: row.get(0),
                person_id: row.get(1),
                timestamp: row.get::<_, chrono::DateTime<chrono::Utc>>(2).to_rfc3339(),
                confidence: row.get(3),
                method: row.get(4),
                photo_path: row.get(5),
            });
        }

        Ok(checkins)
    }
}

fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = calculate_cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        let similarity2 = calculate_cosine_similarity(&c, &d);
        assert!((similarity2 - 0.0).abs() < 0.001);
    }
}