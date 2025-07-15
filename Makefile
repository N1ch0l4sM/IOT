# Makefile para facilitar comandos do projeto

.PHONY: help install test run docker-up docker-down dashboard clean

help:  ## Mostra esta mensagem de ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Instala as dependências do projeto
	pip install -r requirements.txt

test:  ## Executa os testes
	python -m pytest tests/ -v --cov=src

run:  ## Executa o pipeline principal
	python main.py

docker-up:  ## Sobe os serviços Docker
	docker-compose up -d

docker-down:  ## Para os serviços Docker
	docker-compose down

docker-logs:  ## Mostra logs dos serviços Docker
	docker-compose logs -f

dashboard:  ## Inicia o dashboard Streamlit
	streamlit run dashboard/main.py --server.port=8501

airflow-init:  ## Inicializa o banco do Airflow
	docker-compose run --rm airflow-webserver airflow db init

airflow-user:  ## Cria usuário admin do Airflow
	docker-compose run --rm airflow-webserver airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin

setup:  ## Setup inicial completo
	make install
	make docker-up
	sleep 30
	make airflow-init
	make airflow-user

clean:  ## Remove arquivos temporários
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete

format:  ## Formata o código com black
	black src/ tests/ dashboard/

lint:  ## Executa linting com flake8
	flake8 src/ tests/ dashboard/

jupyter:  ## Inicia Jupyter Lab
	jupyter lab notebooks/
