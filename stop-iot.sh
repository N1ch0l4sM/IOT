#!/bin/bash

# Script para parar o ambiente IoT
set -e

echo "🛑 Parando Pipeline IoT..."

# Parar todos os serviços
echo "⏹️ Parando serviços..."
docker-compose down

echo ""
echo "✅ Todos os serviços foram parados!"
echo ""
echo "📊 Opções disponíveis:"
echo "  • Para reiniciar: ./start-iot.sh"
echo "  • Para limpar tudo: docker-compose down -v --remove-orphans"
echo "  • Para ver containers parados: docker-compose ps -a"
