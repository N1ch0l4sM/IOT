# 🏗️ Arquitetura do Sistema

## Visão Geral

O Pipeline IoT para Previsão de Chuvas é um sistema completo de análise de dados climáticos que utiliza tecnologias modernas para coleta, processamento, armazenamento e predição de dados meteorológicos.

## Componentes Principais

### 1. Camada de Dados
- **PostgreSQL**: Banco relacional para dados processados e metadados
- **MinIO**: Data Lake para armazenamento de dados brutos e modelos
- **Redis**: Cache e broker para Airflow

### 2. Camada de Processamento
- **Apache Airflow**: Orquestração de pipelines ETL
- **Python/Pandas**: Processamento e transformação de dados
- **Scikit-learn**: Machine Learning e modelagem

### 3. Camada de Apresentação
- **Streamlit**: Dashboard interativo
- **Plotly**: Visualizações dinâmicas
- **REST API**: Interface para integração (futuro)

### 4. Infraestrutura
- **Docker**: Containerização de serviços
- **Docker Compose**: Orquestração local
- **Nginx**: Reverse proxy (futuro)

## Fluxo de Dados

```
[Fonte de Dados] → [Extração] → [Transformação] → [Carregamento] → [ML] → [Predições] → [Dashboard]
     ↓              ↓             ↓                ↓            ↓         ↓            ↓
   Kaggle       Airflow      Data Cleaning     PostgreSQL   Scikit    Database    Streamlit
   APIs         Python       Feature Eng.      MinIO        Learn     MinIO       Plotly
```

## Detalhamento Técnico

### Pipeline ETL (Airflow)

#### 1. Extração (`extract_weather_data`)
- Coleta dados de fontes externas (Kaggle, APIs)
- Validação inicial de formato
- Armazenamento em MinIO (raw bucket)

#### 2. Transformação (`transform_weather_data`)
- Limpeza de dados (valores nulos, outliers)
- Feature engineering (sensação térmica, índices derivados)
- Validação de qualidade

#### 3. Carregamento (`load_weather_data`)
- Armazenamento em PostgreSQL
- Versionamento no MinIO
- Indexação para consultas rápidas

#### 4. Treinamento ML (`train_ml_model`)
- Retreinamento periódico (diário)
- Validação cruzada
- Métricas de performance
- Versionamento de modelos

#### 5. Predições (`make_predictions`)
- Predições em tempo real
- Armazenamento de resultados
- Cálculo de confiança

### Modelo de Machine Learning

#### Algoritmo
- **Random Forest Classifier**: Robusto e interpretável
- **Features**: 12+ variáveis meteorológicas e temporais
- **Target**: Classificação binária (chuva/não chuva)

#### Features Principais
1. **Meteorológicas**
   - Temperatura, Umidade, Pressão
   - Velocidade e direção do vento
   - Sensação térmica, Ponto de orvalho

2. **Temporais**
   - Hora do dia, Dia da semana
   - Mês, Estação do ano

3. **Derivadas**
   - Interações entre variáveis
   - Tendências temporais
   - Índices compostos

#### Métricas de Avaliação
- **AUC-ROC**: Capacidade de discriminação
- **Precisão/Recall**: Qualidade das predições
- **F1-Score**: Métrica balanceada
- **Matriz de Confusão**: Análise detalhada de erros

### Base de Dados

#### Schema Principal

```sql
-- Tabela de dados climáticos
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    wind_speed FLOAT,
    wind_direction VARCHAR(10),
    precipitation FLOAT,
    feels_like FLOAT,
    dew_point FLOAT,
    recorded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de predições
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    weather_data_id INTEGER REFERENCES weather_data(id),
    prediction BOOLEAN NOT NULL,
    confidence FLOAT,
    model_version VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de métricas do modelo
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    auc FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    accuracy FLOAT,
    trained_at TIMESTAMP DEFAULT NOW()
);
```

#### Índices para Performance
```sql
CREATE INDEX idx_weather_location_time ON weather_data(location, recorded_at);
CREATE INDEX idx_predictions_time ON predictions(predicted_at);
CREATE INDEX idx_weather_precipitation ON weather_data(precipitation);
```

### Dashboard Streamlit

#### Páginas Principais
1. **Análise Climática**
   - Séries temporais de variáveis
   - Distribuições estatísticas
   - Correlações entre variáveis

2. **Predições**
   - Histórico de predições
   - Confiança dos modelos
   - Simulador interativo

3. **Modelo ML**
   - Métricas de performance
   - Importância das features
   - Matriz de confusão

4. **Monitoramento**
   - Status dos serviços
   - Logs do sistema
   - Estatísticas operacionais

#### Tecnologias
- **Streamlit**: Framework web
- **Plotly**: Gráficos interativos
- **Pandas**: Manipulação de dados
- **SQLAlchemy**: ORM para banco

## Configuração e Deploy

### Desenvolvimento Local
```bash
# 1. Clonar repositório
git clone <repo_url>
cd IOT

# 2. Configurar ambiente
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configurar variáveis
cp .env.example .env
nano .env

# 4. Executar
python main.py
```

### Deploy com Docker
```bash
# 1. Subir serviços
docker-compose up -d

# 2. Inicializar Airflow
make airflow-init
make airflow-user

# 3. Verificar status
docker-compose ps
```

### Deploy em Produção (Kubernetes)
```yaml
# Exemplo de deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weather-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: weather-api
  template:
    metadata:
      labels:
        app: weather-api
    spec:
      containers:
      - name: api
        image: weather-iot:latest
        ports:
        - containerPort: 8000
```

## Monitoramento e Observabilidade

### Logs
- **Structured Logging**: JSON formatado
- **Centralização**: ELK Stack (futuro)
- **Níveis**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Métricas
- **Sistema**: CPU, Memória, Disco
- **Aplicação**: Throughput, Latência, Erros
- **Negócio**: Precisão, Cobertura, Drift

### Alertas
- **Falhas de Pipeline**: Email/Slack
- **Degradação de Modelo**: Monitoramento contínuo
- **Indisponibilidade**: Health checks

## Segurança

### Dados
- **Criptografia**: TLS em trânsito, AES em repouso
- **Backup**: Snapshots automáticos
- **Retenção**: Políticas de lifecycle

### Acesso
- **Autenticação**: JWT, OAuth2
- **Autorização**: RBAC
- **Auditoria**: Logs de acesso

### Infraestrutura
- **Network**: VPC, Security Groups
- **Secrets**: Vault, Kubernetes Secrets
- **Compliance**: LGPD, GDPR

## Performance e Escalabilidade

### Otimizações Atuais
- **Índices de banco**: Consultas rápidas
- **Cache**: Redis para dados frequentes
- **Batch Processing**: Processamento em lotes

### Escalabilidade Futura
- **Horizontal Scaling**: Kubernetes HPA
- **Data Partitioning**: Por localização/tempo
- **CDN**: Cache de assets estáticos

## Troubleshooting

### Problemas Comuns

#### 1. Airflow não inicia
```bash
# Verificar logs
docker-compose logs airflow-webserver

# Limpar banco
docker-compose down -v
docker-compose up -d
```

#### 2. Conexão com banco falha
```bash
# Verificar variáveis
env | grep DB_

# Testar conexão
docker-compose exec postgres psql -U postgres
```

#### 3. MinIO inacessível
```bash
# Verificar status
docker-compose ps minio

# Logs
docker-compose logs minio
```

#### 4. Dashboard lento
```bash
# Verificar cache
docker-compose restart redis

# Otimizar queries
EXPLAIN ANALYZE SELECT * FROM weather_data;
```

### Debugging

#### Logs Estruturados
```python
import logging
import json

logger = logging.getLogger(__name__)
logger.info(json.dumps({
    "event": "prediction_made",
    "model_version": "1.0.0",
    "confidence": 0.85,
    "timestamp": "2024-01-01T12:00:00Z"
}))
```

#### Profiling
```python
import cProfile
import pstats

# Profiling de função
pr = cProfile.Profile()
pr.enable()
result = my_function()
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(10)
```

## Roadmap Técnico

### Q1 2024
- [ ] APIs REST para predições
- [ ] Autenticação e autorização
- [ ] Testes de integração

### Q2 2024
- [ ] Deploy em nuvem (AWS/GCP)
- [ ] Monitoramento com Prometheus
- [ ] CI/CD automatizado

### Q3 2024
- [ ] Modelos avançados (XGBoost, Neural Networks)
- [ ] Real-time streaming (Kafka)
- [ ] Mobile app

### Q4 2024
- [ ] Multi-tenancy
- [ ] Edge computing
- [ ] IoT device integration

---

Esta documentação é atualizada regularmente. Para dúvidas técnicas específicas, consulte o código fonte ou abra uma issue no repositório.
