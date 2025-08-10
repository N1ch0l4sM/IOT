#!/bin/bash

# Script para iniciar o ambiente IoT
set -e

echo "🚀 Iniciando Pipeline IoT..."

# Verificar se Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está rodando. Por favor, inicie o Docker primeiro."
    exit 1
fi

# Verificar se docker compose está instalado
if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ docker compose não encontrado. Por favor, instale o docker compose."
    exit 1
fi

# Criar diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p logs plugins

# Definir permissões dos diretórios (com fallback em caso de erro)
echo "🔧 Configurando permissões..."
chmod 755 logs plugins dags src 2>/dev/null || {
    echo "⚠️  Não foi possível alterar permissões. Continuando..."
    echo "   Isso pode acontecer em alguns sistemas. O pipeline ainda funcionará."
}

# Iniciar serviços básicos primeiro (bancos de dados)
echo "🗄️ Iniciando bancos de dados..."
docker compose up -d postgres mongodb redis minio

# Aguardar bancos estarem prontos
echo "⏳ Aguardando bancos ficarem prontos..."
sleep 15

# Verificar se PostgreSQL está pronto
echo "🔍 Verificando PostgreSQL..."
until docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "Aguardando PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL pronto!"

# Verificar se MongoDB está pronto
echo "🔍 Verificando MongoDB..."
until docker compose exec mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
    echo "Aguardando MongoDB..."
    sleep 2
done
echo "✅ MongoDB pronto!"

# Iniciar Airflow Database
echo "🗄️ Iniciando Airflow Database..."
docker compose up -d airflow-db

# Aguardar Airflow DB estar pronto
echo "⏳ Aguardando Airflow DB..."
sleep 10

until docker compose exec airflow-db pg_isready -U airflow > /dev/null 2>&1; do
    echo "Aguardando Airflow DB..."
    sleep 2
done
echo "✅ Airflow DB pronto!"

# Inicializar Airflow (apenas se necessário)
echo "🔧 Inicializando Airflow..."
docker compose run --rm airflow-webserver airflow db init || echo "DB já inicializado"

# Criar usuário admin do Airflow
echo "👤 Criando usuário admin do Airflow..."
docker compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@iot.com \
    --password admin123 || echo "Usuário já existe"

# Iniciar todos os serviços do Airflow
echo "🌬️ Iniciando Airflow..."
docker compose up -d airflow-webserver airflow-scheduler airflow-worker

echo ""
echo "🎉 Ambiente IoT iniciado com sucesso!"
echo ""
echo "📊 Serviços disponíveis:"
echo "  • PostgreSQL (dados estruturados): localhost:5432"
echo "  • MongoDB (dados brutos): localhost:27017"
echo "  • MinIO (armazenamento): localhost:9000 (console: localhost:9001)"
echo "  • Redis (cache): localhost:6379"
echo "  • Airflow (orquestração): http://localhost:8080"
echo "  • Airflow DB: localhost:5433"
echo ""
echo "🔐 Credenciais:"
echo "  • PostgreSQL: postgres/postgres123"
echo "  • MongoDB: admin/admin123"
echo "  • MinIO: minioadmin/minioadmin123"
echo "  • Airflow: admin/admin123"
echo ""
echo "📝 Próximos passos:"
echo "  1. Acesse Airflow em http://localhost:8080"
echo "  2. Execute: python main.py (para teste local)"
echo "  3. Ou use: make run-pipeline"
echo ""
echo "📊 Para verificar status: docker compose ps"
echo "📋 Para ver logs: docker compose logs -f [serviço]"
echo "🛑 Para parar tudo: docker compose down"
