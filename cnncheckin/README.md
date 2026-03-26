# CNN CheckIn - Sistema de Reconhecimento Facial

## 📁 Estrutura do Projeto

```
cnncheckin/
├── src/
│   ├── main.rs              # Arquivo principal com CLI
│   ├── camera.rs            # Captura de webcam
│   ├── cnn_model.rs         # Modelo CNN com Burn
│   ├── database.rs          # PostgreSQL integration
│   ├── face_detector.rs     # Detecção e reconhecimento
│   ├── image_processor.rs   # Processamento de imagens
│   ├── config.rs            # Configurações
│   └── utils.rs             # Utilitários
├── Cargo.toml              # Dependências
├── config.toml             # Configuração (criado automaticamente)
├── dados/
│   ├── fotos_webcam/       # Fotos capturadas
│   ├── fotos_treino/       # Dataset de treinamento
│   ├── modelos/            # Modelos treinados
│   ├── temp/               # Arquivos temporários
│   └── logs/               # Logs do sistema
└── README.md
```

# executar
```
cargo run --release -- init
cargo run --release -- capture
cargo run --release -- train
```

## 🚀 Instalação e Configuração

### 1. Pré-requisitos

#### Sistema Operacional
- **Linux**: Ubuntu 20.04+ (recomendado)
- **Windows**: Windows 10+ com WSL2
- **macOS**: macOS 11+

#### Dependências do Sistema
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libv4l-dev v4l-utils
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y cmake libclang-dev

# Fedora/RHEL
sudo dnf install -y gcc gcc-c++ pkg-config v4l2loopback-dkms
sudo dnf install -y postgresql postgresql-server postgresql-contrib

# macOS (com Homebrew)
brew install postgresql
brew install cmake
```

#### Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup update stable
```

### 2. Configuração do Banco de Dados

```bash
# Iniciar PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Criar usuário e banco
sudo -u postgres psql
```

```sql
CREATE USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE cnncheckin OWNER postgres;
GRANT ALL PRIVILEGES ON DATABASE cnncheckin TO postgres;
\q
```

### 3. Compilação e Execução

```bash
# Clonar e entrar no projeto
git clone <seu-repositorio>
cd cnncheckin

# Compilar (modo debug)
cargo build

# Compilar (modo release - recomendado para produção)
cargo build --release

# Executar
cargo run --release -- --help
```

# Executar
```bash
cargo run --release -- init
```

## 📋 Comandos Disponíveis

### Configuração Inicial
```bash
# Configurar banco de dados
cargo run --release -- database setup

# Verificar configuração
cargo run --release -- database list
```

### Captura de Imagens para Treinamento
```bash
# Capturar 10 fotos por pessoa (padrão)
cargo run --release -- capture

# Capturar número específico de fotos
cargo run --release -- capture --count 15
```

**Controles durante captura:**
- `ESPAÇO` ou `S`: Capturar foto
- `N`: Nova pessoa
- `R`: Reset contador FPS
- `ESC`: Sair

### Treinamento do Modelo
```bash
# Treinar modelo com dataset padrão
cargo run --release -- train

# Especificar diretório de dados
cargo run --release -- train --input-dir ./meu_dataset
```

### Reconhecimento Facial

#### Modo Aprendizado (adicionar novas faces)
```bash
# Tempo real
cargo run --release -- recognize --realtime
# Selecionar: 1) Aprendizado

# Foto única
cargo run --release -- recognize
# Selecionar: 1) Aprendizado
```

#### Modo Reconhecimento (identificar faces)
```bash
# Tempo real
cargo run --release -- recognize --realtime
# Selecionar: 2) Reconhecimento

# Foto única
cargo run --release -- recognize
# Selecionar: 2) Reconhecimento
```

### Gerenciamento do Banco
```bash
# Listar modelos salvos
cargo run --release -- database list

# Exportar modelo específico
cargo run --release -- database export --model-id 1

# Estatísticas
cargo run --release -- database stats
```

## ⚙️ Configuração Avançada

### Arquivo `config.toml`

O sistema cria automaticamente um arquivo de configuração. Você pode editá-lo:

```toml
[database]
host = "localhost"
port = 5432
database = "cnncheckin"
username = "postgres"
password = "postgres"
max_connections = 10

[camera]
device_path = "/dev/video0"
width = 640
height = 480
fps = 30
preferred_formats = ["RGB3", "YUYV"]

[model]
input_size = [3, 128, 128]
batch_size = 32
epochs = 50
learning_rate = 0.001
validation_split = 0.2
early_stopping_patience = 10

[paths]
photos_dir = "../../dados/fotos_webcam"
training_dir = "../../dados/fotos_treino"
models_dir = "../../dados/modelos"
temp_dir = "../../dados/temp"
logs_dir = "../../dados/logs"

[recognition]
confidence_threshold = 0.7
similarity_threshold = 0.8
max_faces_per_frame = 5
face_detection_model = "simple"
```

### Variáveis de Ambiente

```bash
# Logging
export RUST_LOG=info
export RUST_BACKTRACE=1

# Banco de dados (opcional)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/cnncheckin"

# GPU (se disponível)
export WGPU_BACKEND=vulkan  # ou dx12, metal, gl
```

## 🔧 Fluxo de Trabalho Recomendado

### 1. Preparação do Dataset
1. Execute `cargo run --release -- capture`
2. Capture pelo menos 10-15 fotos por pessoa
3. Varie condições: iluminação, ângulos, expressões
4. Nomeie as pessoas claramente

### 2. Treinamento
1. Execute `cargo run --release -- train`
2. Aguarde o treinamento completar (pode demorar)
3. Verifique a acurácia final (>85% é bom)

### 3. Uso em Produção
1. Use `cargo run --release -- recognize --realtime`
2. Selecione modo "Reconhecimento"
3. Sistema identificará faces conhecidas automaticamente

### 4. Adição de Novas Pessoas
1. Use modo "Aprendizado" para adicionar faces
2. Retreine periodicamente para melhor performance

## 🎯 Dicas de Performance

### Hardware Recomendado
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB+ (16GB para datasets grandes)
- **GPU**: Opcional, mas acelera treinamento
- **Webcam**: 720p+ com boa iluminação

### Otimizações
```bash
# Compilação otimizada
cargo build --release

# Com features específicas
cargo build --release --features gpu-acceleration

# Benchmark de performance
cargo bench
```

### Monitoramento
- Logs em `dados/logs/`
- Métricas de FPS na interface
- Acurácia do modelo no banco

## 🐛 Troubleshooting

### Problemas Comuns

#### "Câmera não encontrada"
```bash
# Verificar dispositivos disponíveis
ls /dev/video*
v4l2-ctl --list-devices

# Testar câmera
ffplay /dev/video0
```

#### "Erro de conexão com banco"
```bash
# Verificar status do PostgreSQL
sudo systemctl status postgresql

# Reiniciar serviço
sudo systemctl restart postgresql

# Verificar conexão
psql -h localhost -U postgres -d cnncheckin
```

#### "Erro de compilação Burn"
```bash
# Atualizar dependências
cargo update

# Limpar cache
cargo clean

# Reinstalar com features específicas
cargo build --release --no-default-features --features postgresql
```

#### "Performance baixa"
```bash
# Verificar recursos
htop
nvidia-smi  # Se tiver GPU

# Ajustar configuração
# Reduzir batch_size no config.toml
# Diminuir resolução da câmera
```

#### "Modelo não converge"
- Verifique qualidade das imagens
- Aumente número de épocas
- Ajuste learning rate
- Use data augmentation
- Certifique-se de ter dados balanceados

### Logs e Debug

```bash
# Logs detalhados
RUST_LOG=debug cargo run --release -- train

# Logs específicos por módulo
RUST_LOG=cnncheckin::cnn_model=debug cargo run --release

# Salvar logs em arquivo
cargo run --release 2>&1 | tee execution.log
```

## 📊 Estrutura do Banco de Dados

### Tabelas Principais

#### `models`
```sql
CREATE TABLE models (
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
);
```

#### `persons`
```sql
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    embedding REAL[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE,
    photo_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);
```

#### `checkins`
```sql
CREATE TABLE checkins (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confidence REAL NOT NULL,
    method VARCHAR(50) NOT NULL,
    photo_path TEXT
);
```

### Consultas Úteis

```sql
-- Estatísticas gerais
SELECT 
    (SELECT COUNT(*) FROM models WHERE is_active = TRUE) as active_models,
    (SELECT COUNT(*) FROM persons WHERE is_active = TRUE) as total_persons,
    (SELECT COUNT(*) FROM checkins WHERE DATE(timestamp) = CURRENT_DATE) as checkins_today;

-- Top 10 pessoas mais reconhecidas
SELECT p.name, COUNT(c.id) as checkin_count
FROM persons p
LEFT JOIN checkins c ON p.id = c.person_id
WHERE c.method = 'recognition'
GROUP BY p.name
ORDER BY checkin_count DESC
LIMIT 10;

-- Histórico de checkins recentes
SELECT p.name, c.timestamp, c.confidence, c.method
FROM checkins c
JOIN persons p ON c.person_id = p.id
ORDER BY c.timestamp DESC
LIMIT 50;
```

## 🧪 Testes

### Executar Testes
```bash
# Todos os testes
cargo test

# Testes específicos
cargo test --lib camera
cargo test --lib database

# Testes de integração
cargo test --test integration

# Com logs
cargo test -- --nocapture
```

### Benchmarks
```bash
# Benchmark de performance
cargo bench

# Benchmark específico
cargo bench cnn_performance
```

## 🔒 Segurança e Privacidade

### Considerações Importantes
- **Dados biométricos**: Embeddings faciais são dados sensíveis
- **LGPD/GDPR**: Implemente consentimento e direito ao esquecimento
- **Criptografia**: Considere criptografar embeddings no banco
- **Acesso**: Use autenticação adequada em produção

### Implementações Recomendadas
```bash
# Backup seguro do banco
pg_dump -h localhost -U postgres cnncheckin > backup_$(date +%Y%m%d).sql

# Limpeza periódica de dados antigos
# (implemente via SQL jobs ou cron)
```

## 📈 Monitoramento e Métricas

### Métricas Importantes
- **FPS da câmera**: >20 FPS para boa experiência
- **Tempo de inferência**: <100ms por face
- **Acurácia do modelo**: >85% em validação
- **Taxa de reconhecimento**: >90% faces conhecidas
- **Falsos positivos**: <5%

### Dashboard SQL
```sql
-- Performance recente
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_checkins,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT person_id) as unique_persons
FROM checkins 
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

## 🚀 Deploy e Produção

### Systemd Service
Criar `/etc/systemd/system/cnncheckin.service`:

```ini
[Unit]
Description=CNN CheckIn Recognition System
After=network.target postgresql.service

[Service]
Type=simple
User=cnncheckin
WorkingDirectory=/opt/cnncheckin
ExecStart=/opt/cnncheckin/target/release/cnncheckin recognize --realtime
Restart=always
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable cnncheckin
sudo systemctl start cnncheckin
sudo systemctl status cnncheckin
```

### Docker (Opcional)
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y postgresql-client libv4l-0
COPY --from=builder /app/target/release/cnncheckin /usr/local/bin/
CMD ["cnncheckin", "recognize", "--realtime"]
```

### Backup Automático
```bash
#!/bin/bash
# backup.sh
pg_dump -h localhost -U postgres cnncheckin | \
gzip > /backup/cnncheckin_$(date +%Y%m%d_%H%M%S).sql.gz

# Manter apenas últimos 30 backups
find /backup -name "cnncheckin_*.sql.gz" -mtime +30 -delete
```

## 📚 Referências e Recursos

### Documentação das Dependências
- [Burn ML Framework](https://burn.dev/)
- [PostgreSQL](https://www.postgresql.org/docs/)
- [Rust](https://doc.rust-lang.org/)
- [rscam](https://docs.rs/rscam/)
- [ndarray](https://docs.rs/ndarray/)

### Artigos Relevantes
- Face Recognition with Deep Learning
- CNN Architecture for Face Recognition
- Real-time Face Detection Algorithms

### Contribuindo
1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo `LICENSE` para detalhes.

## 🤝 Suporte
- **Issues**: Reporte bugs no GitHub Issues
- **Discussões**: Use GitHub Discussions para perguntas
- **Email**: suporte@cnncheckin.com

## 🎯 Roadmap Futuro
- [ ] Suporte para múltiplas câmeras
- [ ] Interface web para administração
- [ ] API REST para integração
- [ ] Suporte para reconhecimento em tempo real via streaming
- [ ] Modelos mais avançados (Vision Transformers)
- [ ] Otimizações para edge computing
- [ ] Suporte para detecção de emoções
- [ ] Integração com sistemas de controle de acesso