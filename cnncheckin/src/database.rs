// file: cnncheckin/src/database.rs
// Módulo de integração com PostgreSQL


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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckIn {
    pub id: i32,
    pub person_id: i32,
    pub timestamp: String,
    pub confidence: f32,
}

pub struct Database {
    pool: ConnectionPool,
}

pub struct DatabaseStats {
    pub models: i32,
    pub persons: i32,
    pub checkins_today: i32,
}

impl Database {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = crate::config::Config::load()?;
        let manager = PostgresConnectionManager::new(config.get_database_url().parse()?, R2D2NoTls);
        let pool = Pool::builder().max_size(config.database.max_connections).build(manager)?;
        Ok(Self { pool })
    }

    pub async fn setup_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        client.batch_execute("
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                accuracy REAL NOT NULL,
                num_classes INTEGER NOT NULL,
                num_parameters BIGINT NOT NULL,
                training_epochs INTEGER NOT NULL,
                class_names TEXT[] NOT NULL,
                model_data BYTEA NOT NULL
            );
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                embedding REAL[] NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS checkins (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                confidence REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkins_person_id ON checkins(person_id);
            CREATE INDEX IF NOT EXISTS idx_checkins_timestamp ON checkins(timestamp);
        ")?;
        Ok(())
    }

    pub async fn save_model(&self, model: &TrainedModel) -> Result<i32, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let model_name = format!("model_{}", chrono::Utc::now().timestamp());
        let row = client.query_one(
            "INSERT INTO models (name, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data)
             VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
            &[&model_name, &model.metadata.accuracy, &(model.metadata.num_classes as i32),
              &(model.metadata.num_parameters as i64), &(model.metadata.training_epochs as i32),
              &model.metadata.class_names, &model.weights],
        )?;
        Ok(row.get(0))
    }

    pub async fn load_model(&self, model_id: i32) -> Result<TrainedModel, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let row = client.query_one(
            "SELECT id, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data
             FROM models WHERE id = $1",
            &[&model_id],
        )?;
        Ok(TrainedModel {
            metadata: ModelMetadata {
                id: Some(row.get(0)),
                created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(1).to_rfc3339(),
                accuracy: row.get(2),
                num_classes: row.get::<_, i32>(3) as usize,
                num_parameters: row.get::<_, i64>(4) as usize,
                training_epochs: row.get::<_, i32>(5) as usize,
                class_names: row.get(6),
            },
            weights: row.get(7),
        })
    }

    pub async fn load_latest_model(&self) -> Result<TrainedModel, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let row = client.query_one(
            "SELECT id, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names, model_data
             FROM models ORDER BY created_at DESC LIMIT 1",
            &[],
        )?;
        Ok(TrainedModel {
            metadata: ModelMetadata {
                id: Some(row.get(0)),
                created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(1).to_rfc3339(),
                accuracy: row.get(2),
                num_classes: row.get::<_, i32>(3) as usize,
                num_parameters: row.get::<_, i64>(4) as usize,
                training_epochs: row.get::<_, i32>(5) as usize,
                class_names: row.get(6),
            },
            weights: row.get(7),
        })
    }

    pub async fn list_models(&self) -> Result<Vec<ModelMetadata>, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let rows = client.query(
            "SELECT id, created_at, accuracy, num_classes, num_parameters, training_epochs, class_names
             FROM models ORDER BY created_at DESC",
            &[],
        )?;
        let models = rows.into_iter().map(|row| ModelMetadata {
            id: Some(row.get(0)),
            created_at: row.get::<_, chrono::DateTime<chrono::Utc>>(1).to_rfc3339(),
            accuracy: row.get(2),
            num_classes: row.get::<_, i32>(3) as usize,
            num_parameters: row.get::<_, i64>(4) as usize,
            training_epochs: row.get::<_, i32>(5) as usize,
            class_names: row.get(6),
        }).collect();
        Ok(models)
    }

    pub async fn save_person(&self, name: &str, embedding: &[f32]) -> Result<i32, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let row = client.query_one(
            "INSERT INTO persons (name, embedding) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding RETURNING id",
            &[&name, &embedding],
        )?;
        Ok(row.get(0))
    }

    pub async fn find_similar_person(&self, embedding: &[f32], threshold: f32) -> Result<Option<Person>, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let rows = client.query("SELECT id, name, embedding, created_at FROM persons", &[])?;
        let mut best_match: Option<Person> = None;
        let mut best_similarity = 0.0;

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
                });
            }
        }
        Ok(best_match)
    }

    pub async fn record_checkin(&self, person_id: i32, confidence: f32) -> Result<i32, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let row = client.query_one(
            "INSERT INTO checkins (person_id, confidence) VALUES ($1, $2) RETURNING id",
            &[&person_id, &confidence],
        )?;
        Ok(row.get(0))
    }

    pub async fn get_stats(&self) -> Result<DatabaseStats, Box<dyn std::error::Error>> {
        let mut client = self.pool.get()?;
        let row = client.query_one(
            "SELECT 
                (SELECT COUNT(*) FROM models) as models,
                (SELECT COUNT(*) FROM persons) as persons,
                (SELECT COUNT(*) FROM checkins WHERE DATE(timestamp) = CURRENT_DATE) as checkins_today",
            &[],
        )?;
        Ok(DatabaseStats {
            models: row.get(0),
            persons: row.get(1),
            checkins_today: row.get(2),
        })
    }
}

fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((calculate_cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }
}