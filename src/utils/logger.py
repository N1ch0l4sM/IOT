"""
Utilitários de logging
"""
import logging
import os
from datetime import datetime
from src.config import DATA_PATHS

def setup_logging(level=logging.INFO):
    """Configura logging para o projeto"""
    
    # Criar diretório de logs se não existir
    log_dir = DATA_PATHS['logs']
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Configurar logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configurado")
    
    return logger
