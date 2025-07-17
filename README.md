# Pipeline IoT para Previsão de Chuvas

Um pipeline completo para coletar, processar e analisar dados climáticos em tempo real e prever se vai chover.

## 🌦️ Visão Geral

Este projeto implementa um pipeline de dados IoT completo para previsão de chuvas, incluindo:
- **Coleta de dados** de fontes climáticas (Kaggle)
- **Processamento ETL** com Apache Airflow
- **Armazenamento** em PostgreSQL e MinIO
- **Machine Learning** com Scikit-learn

## 🚀 Stack Tecnológica

- **Dados**: Kaggle Weather Dataset
- **Processamento**: Python, Pandas, Apache Airflow
- **Armazenamento**: PostgreSQL (relacional) + MinIO (Data Lake)
- **Machine Learning**: Scikit-learn (classificação binária)
- **Orquestração**: Docker Compose

## 📁 Estrutura do Projeto

```
IOT/
├── dags/                   # Airflow DAGs
├── data/                   # Dados brutos e processados
├── models/                 # Modelos de ML
├── src/                    # Código fonte
│   ├── data_processing/    # ETL scripts
│   ├── ml/                 # Machine Learning
│   └── utils/              # Utilitários
├── docker/                 # Configurações Docker
├── notebooks/              # Jupyter notebooks
└── tests/                  # Testes
```

## 🔧 Configuração do Ambiente

### Pré-requisitos
- Python 3.10+ (testado com Python 3.12)
- Docker e Docker Compose
- Git

### 1. Instalar dependências do sistema (Ubuntu/Debian)
```bash
# Instalar pip e venv
sudo apt update
sudo apt install python3-pip python3-venv python3-dev

# Instalar Docker (se não estiver instalado)
sudo apt install docker.io docker-compose

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
# Fazer logout e login novamente
```

### 2. Criar ambiente virtual
```bash
cd /home/nicholas/projects/IOT
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependências Python
```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Editar .env com suas configurações
nano .env
```

### 5. Subir serviços com Docker
```bash
# Opção 1: Usar Docker Compose
docker-compose up -d

# Opção 2: Usar Makefile
make docker-up
make setup  # Para setup completo incluindo Airflow
```

### 6. Executar pipeline (desenvolvimento local)
```bash
# Executar pipeline principal
python main.py

# Ou usar Makefile
make run
```

## 🌐 Serviços

- **Airflow**: http://localhost:8080
- **MinIO**: http://localhost:9001
- **PostgreSQL**: localhost:5432

## 📊 Funcionalidades

### Pipeline ETL
- Coleta automática de dados climáticos
- Limpeza e normalização dos dados
- Armazenamento em Data Warehouse

### Machine Learning
- Modelo de classificação binária (chuva/não chuva)
- Treinamento automatizado
- Avaliação de performance


## 🔍 Uso

1. **Dados**: Coloque os dados do Kaggle na pasta `data/raw/`
2. **Treinamento**: Execute o pipeline para treinar o modelo
3. **Previsão**: Use o dashboard para fazer previsões
4. **Monitoramento**: Acompanhe no Airflow

## 🎯 Como Usar

### Desenvolvimento Local
```bash
# 1. Gerar dados de exemplo
python generate_sample_data.py

# 2. Executar pipeline
python main.py
```

### Produção com Docker
```bash
# 1. Subir todos os serviços
make setup

# 2. Acessar interfaces
# - Airflow: http://localhost:8080 (admin/admin)
# - MinIO: http://localhost:9001 (minioadmin/minioadmin123)
```

### Comandos Úteis
```bash
make help          # Ver todos os comandos
make test          # Executar testes
make docker-logs   # Ver logs dos serviços
make clean         # Limpar arquivos temporários
```

## 📊 Estrutura de Dados

### Dados de Entrada
- **Temperatura**: °C
- **Umidade**: %
- **Pressão**: hPa
- **Vento**: velocidade (km/h) e direção
- **Precipitação**: mm
- **Localização**: cidades brasileiras

### Features Derivadas
- **Sensação térmica**
- **Ponto de orvalho**
- **Índice de calor**
- **Tendência de pressão**
- **Features temporais** (hora, dia da semana, mês, estação)

### Target
- **will_rain**: classificação binária (0/1)

## 🔍 Monitoramento

### Métricas do Modelo
- **AUC-ROC**: Área sob a curva ROC
- **Precisão**: Verdadeiros positivos / (VP + Falsos positivos)
- **Recall**: Verdadeiros positivos / (VP + Falsos negativos)
- **F1-Score**: Média harmônica entre precisão e recall

### Pipeline de Dados
- **Qualidade**: Verificação de valores nulos e outliers
- **Latência**: Tempo de processamento
- **Volume**: Quantidade de dados processados
- **Frescar**: Timestamp dos últimos dados

## 📈 Próximos Passos

- [ ] Integração com APIs de clima em tempo real (OpenWeatherMap, INMET)
- [ ] Modelos mais avançados (XGBoost, Deep Learning)
- [ ] Alertas automáticos via email/SMS
- [ ] API REST para previsões
- [ ] Deploy em nuvem (AWS, GCP, Azure)
- [ ] Monitoramento com Prometheus/Grafana
- [ ] CI/CD com GitHub Actions
- [ ] Documentação automática com Sphinx

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/ -v

# Com coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Testes específicos
python -m pytest tests/test_weather_processor.py -v
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Guidelines
- Siga o padrão PEP 8 para Python
- Adicione testes para novas funcionalidades
- Documente novas funções e classes
- Use type hints sempre que possível

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- [Apache Airflow](https://airflow.apache.org/) - Orquestração de workflows
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning
- [PostgreSQL](https://www.postgresql.org/) - Banco de dados
- [MinIO](https://min.io/) - Object Storage
- [Docker](https://www.docker.com/) - Containerização

---

Desenvolvido com ❤️ para a comunidade de Data Science e IoT
