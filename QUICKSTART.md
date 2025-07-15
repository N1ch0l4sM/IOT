# 🚀 Guia de Início Rápido

## Setup Rápido (Linux/Ubuntu)

### 1. Instalar dependências do sistema
```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-dev docker.io docker-compose
sudo usermod -aG docker $USER
# Fazer logout e login novamente
```

### 2. Configurar projeto
```bash
cd /home/nicholas/projects/IOT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Executar com Docker
```bash
# Subir todos os serviços
docker-compose up -d

# Aguardar serviços subirem (30-60 segundos)
# Inicializar Airflow
docker-compose exec airflow-webserver airflow db init
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 4. Acessar aplicações
- **Airflow**: http://localhost:8080 (admin/admin)
- **Dashboard**: http://localhost:8501
- **MinIO**: http://localhost:9001 (minioadmin/minioadmin123)

### 5. Executar pipeline local (opcional)
```bash
source venv/bin/activate
python main.py
```

## Troubleshooting

### Erro: "pip not found"
```bash
sudo apt install python3-pip
```

### Erro: "Docker permission denied"
```bash
sudo usermod -aG docker $USER
# Fazer logout e login novamente
```

### Erro: "Port already in use"
```bash
# Parar serviços existentes
docker-compose down
# Verificar portas
sudo netstat -tulpn | grep :8080
```

### Logs dos serviços
```bash
# Ver logs de todos os serviços
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f airflow-webserver
```
