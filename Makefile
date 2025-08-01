# Makefile para Pipeline IoT
.PHONY: help init install setup-env start stop restart logs clean test run-pipeline

# Configurações
DOCKER_COMPOSE_FILE = docker-compose.yml
PYTHON_ENV = venv
REQUIREMENTS_FILE = requirements-pipeline.txt

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Inicializa projeto criando diretórios necessários
	@echo "🔧 Inicializando Pipeline IoT..."
	@mkdir -p logs plugins data/raw data/processed artifacts models
	@chmod 755 logs plugins dags src 2>/dev/null || echo "⚠️  Permissões não alteradas (isso é normal em alguns sistemas)"
	@echo "✅ Inicialização concluída!"

install: ## Instala dependências Python
	@echo "Instalando dependências..."
	pip install -r $(REQUIREMENTS_FILE)
	@echo "Dependências instaladas com sucesso!"

setup-env: ## Configura ambiente virtual Python
	@echo "Criando ambiente virtual..."
	python -m venv $(PYTHON_ENV)
	@echo "Ativando ambiente virtual e instalando dependências..."
	$(PYTHON_ENV)/bin/pip install --upgrade pip
	$(PYTHON_ENV)/bin/pip install -r $(REQUIREMENTS_FILE)
	@echo "Ambiente configurado! Ative com: source $(PYTHON_ENV)/bin/activate"

start: ## Inicia todos os serviços Docker
	@echo "Iniciando serviços Docker..."
	./start-iot.sh

stop: ## Para todos os serviços Docker
	@echo "Parando serviços Docker..."
	./stop-iot.sh

start-basic: ## Inicia apenas serviços básicos (sem Airflow)
	@echo "Iniciando serviços básicos..."
	docker-compose up -d postgres mongodb minio redis
	@echo "Serviços básicos iniciados!"
	@echo "PostgreSQL: localhost:5432"
	@echo "MongoDB: localhost:27017"
	@echo "MinIO: localhost:9000 (UI: localhost:9001)"
	@echo "Redis: localhost:6379"

restart: stop start ## Reinicia todos os serviços

logs: ## Mostra logs dos serviços
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

logs-postgres: ## Mostra logs do PostgreSQL
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f postgres

logs-mongodb: ## Mostra logs do MongoDB
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f mongodb

logs-minio: ## Mostra logs do MinIO
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f minio

status: ## Mostra status dos serviços
	docker-compose -f $(DOCKER_COMPOSE_FILE) ps

clean: ## Remove containers, volumes e images
	@echo "Limpando ambiente Docker..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down -v --remove-orphans
	docker system prune -f
	@echo "Ambiente limpo!"

clean-logs: ## Remove arquivos de log
	@echo "Removendo logs..."
	rm -rf logs/*.log
	@echo "Logs removidos!"

setup-db: ## Configura banco de dados inicial
	@echo "Configurando banco de dados..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres psql -U postgres -d iot_weather_db -c "SELECT version();"
	@echo "Banco de dados configurado!"

run-pipeline: ## Executa pipeline IoT localmente
	@echo "Executando pipeline IoT..."
	python main.py
	@echo "Pipeline executado!"

run-fetch: ## Executa apenas coleta de dados
	@echo "Executando coleta de dados..."
	python -m src.weather_fetch
	@echo "Coleta executada!"

run-aggregate: ## Executa apenas agregação horária
	@echo "Executando agregação horária..."
	python -m src.weather_hour
	@echo "Agregação executada!"

test: ## Executa testes
	@echo "Executando testes..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Testes executados!"

test-unit: ## Executa apenas testes unitários
	@echo "Executando testes unitários..."
	pytest tests/test_*.py -v
	@echo "Testes unitários executados!"

lint: ## Executa verificação de código
	@echo "Verificando código..."
	flake8 src/ --max-line-length=100
	black --check src/
	@echo "Verificação concluída!"

format: ## Formata código Python
	@echo "Formatando código..."
	black src/
	isort src/
	@echo "Código formatado!"

backup-db: ## Faz backup do banco de dados
	@echo "Fazendo backup do banco..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres pg_dump -U postgres iot_weather_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup criado!"

monitor: ## Mostra monitoramento dos recursos
	@echo "Monitoramento de recursos:"
	docker stats --no-stream

build: ## Reconstrói images Docker
	@echo "Reconstruindo images..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) build --no-cache
	@echo "Images reconstruídas!"

shell-postgres: ## Acessa shell do PostgreSQL
	docker-compose -f $(DOCKER_COMPOSE_FILE) exec postgres psql -U postgres -d iot_weather_db

shell-mongodb: ## Acessa shell do MongoDB
	docker-compose -f $(DOCKER_COMPOSE_FILE) exec mongodb mongo mongo_iot_weather -u admin -p admin123

notebook: ## Inicia Jupyter Notebook
	@echo "Iniciando Jupyter Notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

requirements: ## Atualiza arquivo de requirements
	@echo "Atualizando requirements..."
	pip freeze > requirements-current.txt
	@echo "Requirements atualizados em requirements-current.txt"

# Comandos para desenvolvimento
dev-setup: setup-env start ## Configuração completa para desenvolvimento
	@echo "Ambiente de desenvolvimento configurado!"
	@echo "Próximos passos:"
	@echo "1. source $(PYTHON_ENV)/bin/activate"
	@echo "2. make run-pipeline"

dev-reset: clean setup-env start ## Reset completo do ambiente
	@echo "Ambiente resetado!"
