"""
Configurações principais do projeto
"""
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações do banco de dados
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'weather_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123')
}

# Configurações do MinIO
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
    'bucket': os.getenv('MINIO_BUCKET', 'weather-data')
}

# Configurações do modelo
MODEL_CONFIG = {
    'path': os.getenv('MODEL_PATH', 'models/'),
    'name': os.getenv('MODEL_NAME', 'rain_prediction_model'),
    'version': '1.0.0'
}

# Configurações de API
API_CONFIG = {
    'weather_api_key': os.getenv('WEATHER_API_KEY', ''),
    'kaggle_username': os.getenv('KAGGLE_USERNAME', ''),
    'kaggle_key': os.getenv('KAGGLE_KEY', '')
}

# Caminhos dos dados
DATA_PATHS = {
    'raw': 'data/raw/',
    'processed': 'data/processed/',
    'models': 'models/',
    'logs': 'logs/'
}

# Configurações do Streamlit
STREAMLIT_CONFIG = {
    'port': int(os.getenv('STREAMLIT_PORT', 8501)),
    'title': 'Pipeline IoT - Previsão de Chuvas',
    'page_icon': '🌦️'
}
