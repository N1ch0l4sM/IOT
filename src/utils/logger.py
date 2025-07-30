"""
Utilitários de logging
"""
import logging
import os
from datetime import datetime
from pathlib import Path
import tempfile

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configura o sistema de logging do projeto
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger configurado
    """
    try:
        # Tentar usar diretório do projeto
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_filepath = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Testar se consegue escrever
        test_file = log_filepath.with_suffix('.test')
        test_file.touch()
        test_file.unlink()
        
    except (PermissionError, OSError):
        # Fallback para diretório temporário
        log_dir = Path(tempfile.gettempdir()) / "iot_pipeline_logs"
        log_dir.mkdir(exist_ok=True)
        log_filepath = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        print(f"Aviso: Usando diretório temporário para logs: {log_dir}")
    
    # Configurar formato do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_filepath, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("IOT_Pipeline")
    logger.info(f"Sistema de logging configurado - Arquivo: {log_filepath}")
    
    return logger
