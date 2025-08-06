// projeto: lstmfiletrain
// file: src/neural/data.rs

// Importações necessárias
use postgres::Client;
use chrono::NaiveDate;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Axis};
// Adiciona println! e info! para logging/debugging
use log::info;

// Importa TrainingError do local correto dentro do crate
use crate::neural::utils::TrainingError;

// Estrutura para representar um registro de ações
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockRecord {
    pub date: NaiveDate,
    pub opening: f32,
    pub closing: f32,
    pub high: f32,
    pub low: f32,
    pub volume: f32,
    pub variation: f32,
}

// Estrutura para armazenar estatísticas das features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_means: Vec<f32>,
    pub feature_stds: Vec<f32>,
    pub feature_names: Vec<String>,
    pub closing_mean: f32,
    pub closing_std: f32,
}

// Carregador de dados com conexão ao banco
pub struct DataLoader<'a> {
    client: &'a mut Client,
}

impl<'a> DataLoader<'a> {
    // Constrói um novo DataLoader a partir de um cliente PostgreSQL
    pub fn new(client: &'a mut Client) -> Result<Self, TrainingError> {
        Ok(DataLoader { client })
    }

    // Carrega dados de um ativo específico do banco de dados
    pub fn load_asset_data(&mut self, asset: &str) -> Result<Vec<StockRecord>, TrainingError> {
        println!("    📥 DataLoader: Procurando dados para '{}' usando LIKE '%{}%'", asset, asset);
        // Note: The query uses 'LIKE' to match assets stored as 'WEGE3 Dados Históricos'
        let query = "SELECT date, opening, closing, high, low, volume, variation FROM stock_records WHERE asset LIKE $1 ORDER BY date ASC";
        let rows = self.client.query(query, &[&format!("%{}%", asset)])
            .map_err(|e| {
                let err_msg = format!("Erro ao carregar dados do ativo {}: {}", asset, e);
                eprintln!("    ❌ DataLoader Erro: {}", err_msg);
                info!("{}", err_msg); // Mantém log::info também
                TrainingError::DatabaseError(e.to_string())
            })?;

        println!("    ✅ DataLoader: {} linhas encontradas para '{}'", rows.len(), asset);
        if rows.is_empty() {
             println!("    ⚠️  DataLoader: Nenhum dado encontrado para '{}' após a consulta.", asset);
             // Retornar um vetor vazio é aceitável, o chamador verifica isso.
        }

        let mut records = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            match NaiveDate::parse_from_str(row.get::<_, &str>(0), "%d.%m.%Y") {
                Ok(parsed_date) => {
                    records.push(StockRecord {
                        date: parsed_date,
                        opening: row.get(1),
                        closing: row.get(2),
                        high: row.get(3),
                        low: row.get(4),
                        volume: row.get(5),
                        variation: row.get(6),
                    });
                     // Printa amostra dos primeiros registros para debug
                     if i < 3 {
                         let r = records.last().unwrap();
                         println!("      Dado amostra {}: {} - Abertura: {}, Fechamento: {}", i+1, r.date, r.opening, r.closing);
                     }
                }
                Err(e) => {
                    let err_msg = format!("Erro ao parsear data '{}' para o ativo {}: {}", row.get::<_, &str>(0), asset, e);
                    eprintln!("    ❌ DataLoader Erro: {}", err_msg);
                    info!("{}", err_msg);
                    // Em vez de falhar todo o processo, podemos pular registros inválidos
                    // ou parar se for um problema crítico. Aqui, vamos retornar um erro.
                    return Err(TrainingError::DataProcessing(e.to_string()));
                }
            }
        }

        println!("    ✅ DataLoader: {} registros processados com sucesso para '{}'", records.len(), asset);
        info!("Sucesso: Carregados {} registros para o ativo {}", records.len(), asset);
        Ok(records)
    }

    // NOVO: Carrega uma lista de todos os ativos únicos presentes na tabela
    pub fn load_all_assets(&mut self) -> Result<Vec<String>, TrainingError> {
        println!("  📋 DataLoader: Carregando lista de todos os ativos...");
        let query = "SELECT DISTINCT asset FROM stock_records";
        let rows = self.client.query(query, &[])
            .map_err(|e| {
                let err_msg = format!("Erro ao carregar lista de ativos: {}", e);
                eprintln!("  ❌ DataLoader Erro: {}", err_msg);
                info!("{}", err_msg);
                TrainingError::DatabaseError(e.to_string())
            })?;

        let assets: Vec<String> = rows.iter()
            .map(|row| row.get(0))
            .collect();

        println!("  ✅ DataLoader: Encontrados {} ativos únicos", assets.len());
        if assets.is_empty() {
            println!("  ⚠️  DataLoader: Nenhum ativo encontrado no banco de dados.");
        } else {
             println!("  📋 DataLoader: Primeiros 5 ativos encontrados: {:?}", &assets[..std::cmp::min(5, assets.len())]);
        }
        info!("Sucesso: Encontrados {} ativos únicos", assets.len());
        Ok(assets)
    }


    // Cria sequências de dados para treinamento a partir dos registros
    pub fn create_sequences(
        &self,
        records: &[StockRecord],
        seq_length: usize,
    ) -> Result<(Vec<Array2<f32>>, Vec<f32>, FeatureStats), TrainingError> {
        println!("    🔧 Criando sequências com comprimento {} para {} registros...", seq_length, records.len());
        // Verifica se há dados suficientes
        if records.len() <= seq_length {
            let err_msg = format!("Dados insuficientes: {} registros, necessário mais que {}", records.len(), seq_length);
            println!("    ❌ {}", err_msg);
            return Err(TrainingError::DataProcessing(err_msg));
        }

        // Extrai todas as features em um vetor plano
        let features: Vec<f32> = records.iter()
            .flat_map(|r| vec![r.opening, r.closing, r.high, r.low, r.volume, r.variation])
            .collect();
        let feature_names = vec![
            "opening".to_string(),
            "closing".to_string(),
            "high".to_string(),
            "low".to_string(),
            "volume".to_string(),
            "variation".to_string(),
        ];
        let num_features = feature_names.len();

        // Verifica consistência dos dados
        if features.len() % num_features != 0 {
            let err_msg = format!("Número incorreto de features: {} (deve ser múltiplo de {})", features.len(), num_features);
            println!("    ❌ {}", err_msg);
            return Err(TrainingError::DataProcessing(err_msg));
        }

        // Cria a matriz de features
        let feature_matrix = Array2::from_shape_vec((records.len(), num_features), features)
            .map_err(|e| {
                let err_msg = format!("Failed to create feature matrix: {}", e);
                println!("    ❌ {}", err_msg);
                info!("Erro ao criar matriz de features: {}", e);
                TrainingError::DataProcessing(err_msg)
            })?;

        println!("    ✅ Matriz de features criada: {:?} (linhas x colunas)", feature_matrix.dim());

        // Calcula as médias das features
        let feature_means = match feature_matrix.mean_axis(Axis(0)) {
            Some(means) => {
                let means_vec = means.to_vec();
                println!("    📊 Médias das features calculadas: {:?}", means_vec);
                means_vec
            },
            None => {
                let msg = format!("Aviso: Nenhuma média calculada, usando zeros para {} features", num_features);
                println!("    ⚠️  {}", msg);
                info!("{}", msg);
                Array1::zeros(num_features).to_vec()
            },
        };

        // Calcula os desvios padrão das features (CORRIGIDO)
        // std_axis retorna diretamente o Array, não um Option
        let feature_stds = feature_matrix.std_axis(Axis(0), 0.0).to_vec();
        println!("    📊 Desvios padrão das features calculados: {:?}", feature_stds);

        // Extrai estatísticas específicas do preço de fechamento
        let closing_mean = feature_means[num_features - 1]; // Assumindo 'closing' é a última
        let closing_std = feature_stds[num_features - 1];
        println!("    📊 Estatísticas do Fechamento - Média: {:.4}, Desvio Padrão: {:.4}", closing_mean, closing_std);

        // Verificação de segurança para desvio padrão
        if closing_std.abs() < 1e-8 {
            let err_msg = "Desvio padrão do preço de fechamento é muito próximo de zero".to_string();
            println!("    ❌ {}", err_msg);
            return Err(TrainingError::DataProcessing(err_msg));
        }

        // Cria as sequências e targets
        let mut sequences = Vec::new();
        let mut targets = Vec::new();
        let num_sequences = records.len() - seq_length;
        println!("    🔢 Número de sequências a serem criadas: {}", num_sequences);

        for i in 0..num_sequences {
            let seq_slice = feature_matrix.slice(ndarray::s![i..i + seq_length, ..]).to_owned();
            let target = records[i + seq_length].closing;
            
            // --- CORREÇÃO: Clonar para debug antes de mover ---
            // Printa amostra das primeiras sequências para debug (antes de mover)
            if i < 2 {
                 println!("      Sequência amostra {}: Target = {:.4}", i+1, target);
                 // Clona o slice para debug (ou faz o debug antes de mover)
                 let debug_slice = seq_slice.clone(); 
                 // Printa o último passo da sequência para ver os inputs
                 // Ajusta o acesso ao slice para pegar a última linha corretamente
                 if debug_slice.dim().0 > 0 { // Verifica se há linhas
                    let last_row_index = debug_slice.dim().0 - 1;
                    let last_step_data = debug_slice.slice(ndarray::s![last_row_index, ..]).to_vec();
                    // Garante que pegamos apenas os dados das features
                    if last_step_data.len() >= num_features {
                        let last_step_features: Vec<f32> = last_step_data[0..num_features].to_vec();
                        println!("        Último passo da sequência {}: {:?}", i+1, last_step_features);
                    } else {
                         println!("        ⚠️  Dados insuficientes no último passo da sequência {}", i+1);
                    }
                 } else {
                      println!("        ⚠️  Sequência {} está vazia", i+1);
                 }
            }
            // --- FIM CORREÇÃO ---
            
            sequences.push(seq_slice); // Move o seq_slice original para o vetor
            targets.push(target);
        }

        // Cria o objeto de estatísticas
        let feature_stats = FeatureStats {
            feature_means,
            feature_stds,
            feature_names,
            closing_mean,
            closing_std,
        };

        println!("    ✅ Criadas {} sequências com comprimento {}", sequences.len(), seq_length);
        info!("Sucesso: Criadas {} sequências com comprimento {}", sequences.len(), seq_length);
        Ok((sequences, targets, feature_stats))
    }
}