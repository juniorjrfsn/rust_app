// projeto cnncheckin
// src/database.rs — Integração com PostgreSQL

use r2d2::Pool;
use r2d2_postgres::{postgres::NoTls, PostgresConnectionManager};
use serde::{Deserialize, Serialize};

use crate::cnn_model::{cosine_similarity, ModelMetadata, TrainedModel};
use crate::config::DatabaseConfig;
use crate::error::{AppError, Result};

// ────────────────────────────────────────────────
//  Tipos de domínio
// ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i32,
    pub name: String,
    pub embedding: Vec<f32>,
    pub created_at: String,
    pub photo_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckIn {
    pub id: i32,
    pub person_id: i32,
    pub person_name: String,
    pub timestamp: String,
    pub confidence: f32,
    pub method: String,
}

pub struct DatabaseStats {
    pub total_models: i64,
    pub total_persons: i64,
    pub checkins_today: i64,
}

// ────────────────────────────────────────────────
//  Database
// ────────────────────────────────────────────────

#[derive(Clone)]
pub struct Database {
    pool: Pool<PostgresConnectionManager<NoTls>>,
}

impl Database {
    pub fn new(config: &DatabaseConfig) -> Result<Self> {
        let host = config.host.as_deref().unwrap_or("localhost");
        let port = config.port.unwrap_or(5432);
        let user = config.username.as_deref().unwrap_or("postgres");
        let password = config.password.as_deref().unwrap_or("postgres");
        let dbname = config.database.as_deref().unwrap_or("cnncheckin");

        let conn_str = format!(
            "host={} port={} user={} password={} dbname={}",
            host, port, user, password, dbname
        );

        let manager = PostgresConnectionManager::new(
            conn_str.parse().map_err(|e: r2d2_postgres::postgres::Error| AppError::Database(e))?,
            NoTls,
        );
        
        let pool = r2d2::Pool::builder()
            .max_size(config.max_connections.unwrap_or(10))
            .build(manager)
            .map_err(AppError::Pool)?;

        let db = Self { pool };
        db.setup_tables()?;
        Ok(db)
    }

    // ── Configuração ────────────────────────────

    pub fn setup_tables(&self) -> Result<()> {
        let mut conn = self.pool.get()?;
        conn.batch_execute(
            "
            CREATE TABLE IF NOT EXISTS models (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                accuracy    REAL NOT NULL DEFAULT 0.0,
                num_classes INTEGER NOT NULL DEFAULT 0,
                epochs      INTEGER NOT NULL DEFAULT 0,
                k_neighbors INTEGER NOT NULL DEFAULT 5,
                class_names TEXT NOT NULL DEFAULT '[]',
                prototypes  TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS persons (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                embedding   TEXT NOT NULL,
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen   TIMESTAMP,
                photo_count INTEGER NOT NULL DEFAULT 0,
                is_active   INTEGER NOT NULL DEFAULT 1
            );

            ALTER TABLE models ADD COLUMN IF NOT EXISTS epochs INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE models ADD COLUMN IF NOT EXISTS k_neighbors INTEGER NOT NULL DEFAULT 5;
            ALTER TABLE models ADD COLUMN IF NOT EXISTS class_names TEXT NOT NULL DEFAULT '[]';
            ALTER TABLE models ADD COLUMN IF NOT EXISTS prototypes TEXT NOT NULL DEFAULT '[]';
            ALTER TABLE models ADD COLUMN IF NOT EXISTS num_parameters BIGINT NOT NULL DEFAULT 0;
            ALTER TABLE models ALTER COLUMN num_parameters SET DEFAULT 0;
            ALTER TABLE models ALTER COLUMN num_parameters TYPE BIGINT USING num_parameters::bigint;
            UPDATE models SET num_parameters = 0 WHERE num_parameters IS NULL;

            ALTER TABLE models ADD COLUMN IF NOT EXISTS training_epochs BIGINT NOT NULL DEFAULT 0;
            ALTER TABLE models ALTER COLUMN training_epochs SET DEFAULT 0;
            ALTER TABLE models ALTER COLUMN training_epochs TYPE BIGINT USING training_epochs::bigint;
            UPDATE models SET training_epochs = epochs WHERE training_epochs IS NULL;

            ALTER TABLE models ADD COLUMN IF NOT EXISTS model_data BYTEA NOT NULL DEFAULT ''::bytea;
            ALTER TABLE models ALTER COLUMN model_data SET DEFAULT ''::bytea;
            ALTER TABLE models ALTER COLUMN model_data TYPE BYTEA USING model_data::bytea;
            UPDATE models SET model_data = ''::bytea WHERE model_data IS NULL;

            -- Força conversão de tipos antigos (ex: text[] mantenho compatibilidade)
            ALTER TABLE models ALTER COLUMN class_names TYPE TEXT USING class_names::text;
            ALTER TABLE models ALTER COLUMN prototypes TYPE TEXT USING prototypes::text;

            ALTER TABLE persons ADD COLUMN IF NOT EXISTS is_active INTEGER NOT NULL DEFAULT 1;

            CREATE TABLE IF NOT EXISTS checkins (
                id          SERIAL PRIMARY KEY,
                person_id   INTEGER REFERENCES persons(id),
                timestamp   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                confidence  REAL NOT NULL,
                method      TEXT NOT NULL DEFAULT 'recognition',
                notes       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_checkins_person   ON checkins(person_id);
            CREATE INDEX IF NOT EXISTS idx_checkins_ts       ON checkins(timestamp);
            CREATE INDEX IF NOT EXISTS idx_persons_name      ON persons(name);
            ",
        ).map_err(AppError::Database)?;
        Ok(())
    }

    // ── Modelos ─────────────────────────────────

    pub fn save_model(&self, model: &TrainedModel) -> Result<i32> {
        let class_names_json = serde_json::to_string(&model.metadata.class_names)
            .map_err(|e| AppError::Generic(e.to_string()))?;
        let prototypes_json = serde_json::to_string(&model.prototypes)
            .map_err(|e| AppError::Generic(e.to_string()))?;

        let mut conn = self.pool.get()?;
        let row = conn.query_one(
            "INSERT INTO models (name, accuracy, num_classes, epochs, training_epochs, k_neighbors, class_names, prototypes, num_parameters, model_data)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
             ON CONFLICT (name) DO UPDATE SET
               accuracy = EXCLUDED.accuracy,
               num_classes = EXCLUDED.num_classes,
               epochs = EXCLUDED.epochs,
               training_epochs = EXCLUDED.training_epochs,
               k_neighbors = EXCLUDED.k_neighbors,
               class_names = EXCLUDED.class_names,
               prototypes = EXCLUDED.prototypes,
               num_parameters = EXCLUDED.num_parameters,
               model_data = EXCLUDED.model_data
             RETURNING id",
            &[
                &model.metadata.name,
                &model.metadata.accuracy,
                &(model.metadata.num_classes as i32),
                &(model.metadata.training_epochs as i32),
                &(model.metadata.training_epochs as i64),
                &(model.metadata.k_neighbors as i32),
                &class_names_json,
                &prototypes_json,
                &0i64,
                &Vec::<u8>::new(),
            ],
        )?;

        let id: i32 = row.get(0);
        Ok(id)
    }

    pub fn load_model(&self, model_id: i32) -> Result<TrainedModel> {
        let mut conn = self.pool.get()?;
        let row = conn.query_opt(
            "SELECT id, name, created_at, accuracy, num_classes, epochs, training_epochs, k_neighbors, class_names, prototypes
             FROM models WHERE id = $1",
            &[&model_id],
        )?;

        let row = row.ok_or_else(|| AppError::Generic(format!("Modelo {} não encontrado", model_id)))?;

        let id: i32 = row.get(0);
        let name: String = row.get(1);
        let created_at: chrono::DateTime<chrono::Utc> = row.get(2);
        let accuracy: f32 = row.get(3);
        let num_classes: i32 = row.get(4);
        let _epochs: i32 = row.get(5);
        let training_epochs: i64 = row.get(6);
        let k: i32 = row.get(7);
        let class_names_json: String = row.get(8);
        let proto_json: String = row.get(9);

        let class_names: Vec<String> = serde_json::from_str(&class_names_json)
            .map_err(|e| AppError::Generic(e.to_string()))?;
        let prototypes: Vec<(Vec<f32>, usize)> = serde_json::from_str(&proto_json)
            .map_err(|e| AppError::Generic(e.to_string()))?;

        Ok(TrainedModel {
            metadata: ModelMetadata {
                id: Some(id),
                name,
                created_at: created_at.naive_utc().to_string(),
                accuracy,
                num_classes: num_classes as usize,
                training_epochs: training_epochs as usize,
                class_names,
                k_neighbors: k as usize,
            },
            prototypes,
        })
    }

    pub fn load_latest_model(&self) -> Result<TrainedModel> {
        let mut conn = self.pool.get()?;
        let row = conn.query_opt("SELECT id FROM models ORDER BY id DESC LIMIT 1", &[])?;
        let row = row.ok_or_else(|| AppError::Generic("Nenhum modelo treinado no banco.".into()))?;
        let id: i32 = row.get(0);
        self.load_model(id)
    }

    pub fn list_models(&self) -> Result<Vec<ModelMetadata>> {
        let mut conn = self.pool.get()?;
        let rows = conn.query(
            "SELECT id, name, created_at, accuracy, num_classes, epochs, training_epochs, k_neighbors, class_names
             FROM models ORDER BY id DESC",
            &[],
        )?;

        let mut result = Vec::new();
        for row in rows {
            let id: i32 = row.get(0);
            let name: String = row.get(1);
            let created_at: chrono::NaiveDateTime = row.get(2);
            let accuracy: f32 = row.get(3);
            let num_classes: i32 = row.get(4);
            let _epochs: i32 = row.get(5);
            let training_epochs: i64 = row.get(6);
            let k: i32 = row.get(7);
            let class_names_json: String = row.get(8);

            let class_names: Vec<String> = serde_json::from_str(&class_names_json).unwrap_or_default();
            result.push(ModelMetadata {
                id: Some(id),
                name,
                created_at: created_at.to_string(),
                accuracy,
                num_classes: num_classes as usize,
                training_epochs: training_epochs as usize,
                class_names,
                k_neighbors: k as usize,
            });
        }
        Ok(result)
    }

    // ── Pessoas ─────────────────────────────────

    pub fn save_person(&self, name: &str, embedding: &[f32]) -> Result<i32> {
        let embedding_json = serde_json::to_string(embedding)
            .map_err(|e| AppError::Generic(e.to_string()))?;

        let mut conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO persons (name, embedding, photo_count)
             VALUES ($1, $2, 1)
             ON CONFLICT (name) DO UPDATE SET
               embedding   = EXCLUDED.embedding,
               last_seen   = CURRENT_TIMESTAMP,
               photo_count = persons.photo_count + 1",
            &[&name, &embedding_json],
        )?;

        let row = conn.query_one("SELECT id FROM persons WHERE name = $1", &[&name])?;
        let id: i32 = row.get(0);
        Ok(id)
    }

    pub fn find_similar_person(&self, embedding: &[f32], threshold: f32) -> Result<Option<Person>> {
        let mut conn = self.pool.get()?;
        let rows = conn.query(
            "SELECT id, name, embedding, created_at, photo_count FROM persons WHERE is_active = 1",
            &[],
        )?;

        let mut best: Option<(f32, Person)> = None;
        for row in rows {
            let id: i32 = row.get(0);
            let name: String = row.get(1);
            let emb_json: String = row.get(2);
            let created_at: chrono::DateTime<chrono::Utc> = row.get(3);
            let photo_count: i32 = row.get(4);

            if let Ok(emb) = serde_json::from_str::<Vec<f32>>(&emb_json) {
                let sim = cosine_similarity(embedding, &emb);
                if sim >= threshold {
                    if best.as_ref().map_or(true, |(best_sim, _)| sim > *best_sim) {
                        best = Some((sim, Person {
                            id,
                            name,
                            embedding: emb,
                            created_at: created_at.naive_utc().to_string(),
                            photo_count,
                        }));
                    }
                }
            }
        }

        Ok(best.map(|(_, p)| p))
    }

    pub fn list_persons(&self) -> Result<Vec<Person>> {
        let mut conn = self.pool.get()?;
        let rows = conn.query(
            "SELECT id, name, embedding, created_at, photo_count FROM persons WHERE is_active = 1 ORDER BY name",
            &[],
        )?;

        let mut persons = Vec::new();
        for row in rows {
            let id: i32 = row.get(0);
            let name: String = row.get(1);
            let emb_json: String = row.get(2);
            let created_at: chrono::DateTime<chrono::Utc> = row.get(3);
            let photo_count: i32 = row.get(4);

            if let Ok(embedding) = serde_json::from_str::<Vec<f32>>(&emb_json) {
                persons.push(Person {
                    id,
                    name,
                    embedding,
                    created_at: created_at.naive_utc().to_string(),
                    photo_count,
                });
            }
        }
        Ok(persons)
    }

    // ── Check-ins ────────────────────────────────

    pub fn record_checkin(&self, person_id: i32, confidence: f32, method: &str) -> Result<i32> {
        let mut conn = self.pool.get()?;
        let row = conn.query_one(
            "INSERT INTO checkins (person_id, confidence, method) VALUES ($1, $2, $3) RETURNING id",
            &[&person_id, &confidence, &method],
        )?;

        conn.execute(
            "UPDATE persons SET last_seen = CURRENT_TIMESTAMP WHERE id = $1",
            &[&person_id],
        )?;

        let id: i32 = row.get(0);
        Ok(id)
    }

    pub fn list_checkins(&self, limit: usize) -> Result<Vec<CheckIn>> {
        let mut conn = self.pool.get()?;
        let rows = conn.query(
            "SELECT c.id, c.person_id, p.name, c.timestamp, c.confidence, c.method
             FROM checkins c JOIN persons p ON c.person_id = p.id
             ORDER BY c.timestamp DESC LIMIT $1",
            &[&(limit as i64)],
        )?;

        let mut checkins = Vec::new();
        for row in rows {
            let id: i32 = row.get(0);
            let person_id: i32 = row.get(1);
            let person_name: String = row.get(2);
            let timestamp: chrono::DateTime<chrono::Utc> = row.get(3);
            let confidence: f32 = row.get(4);
            let method: String = row.get(5);

            checkins.push(CheckIn {
                id,
                person_id,
                person_name,
                timestamp: timestamp.naive_utc().to_string(),
                confidence,
                method,
            });
        }
        Ok(checkins)
    }

    // ── Estatísticas ─────────────────────────────

    pub fn get_stats(&self) -> Result<DatabaseStats> {
        let mut conn = self.pool.get()?;
        let row = conn.query_one("SELECT COUNT(*) FROM models", &[])?;
        let total_models: i64 = row.get(0);

        let row = conn.query_one("SELECT COUNT(*) FROM persons WHERE is_active = 1", &[])?;
        let total_persons: i64 = row.get(0);

        let row = conn.query_one(
            "SELECT COUNT(*) FROM checkins WHERE timestamp >= current_date",
            &[],
        )?;
        let checkins_today: i64 = row.get(0);

        Ok(DatabaseStats {
            total_models,
            total_persons,
            checkins_today,
        })
    }

    pub fn describe_models_table(&self) -> Result<Vec<(String, String, String)>> {
        let mut conn = self.pool.get()?;
        let rows = conn.query(
            "SELECT column_name, data_type, udt_name FROM information_schema.columns WHERE table_name = 'models' ORDER BY ordinal_position",
            &[],
        )?;

        let mut columns = Vec::new();
        for row in rows {
            let name: String = row.get(0);
            let data_type: String = row.get(1);
            let udt_name: String = row.get(2);
            columns.push((name, data_type, udt_name));
        }
        Ok(columns)
    }
}