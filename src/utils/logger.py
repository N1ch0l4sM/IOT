"""
Sistema de logging configurável para o pipeline IoT
"""
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configura o sistema de logging do projeto com fallback automático
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger configurado com handlers apropriados
        
    Raises:
        Exception: Se não conseguir configurar nenhum sistema de logging
    """
    # Tentar múltiplas localizações para logs
    possible_log_dirs = [
        Path(__file__).parent.parent.parent / "logs",  # Diretório do projeto
        Path.home() / ".iot_pipeline" / "logs",        # Diretório do usuário
        Path(tempfile.gettempdir()) / "iot_pipeline"   # Diretório temporário
    ]
    
    log_filepath = None
    log_dir = None
    
    # Tentar cada diretório até encontrar um que funcione
    for potential_dir in possible_log_dirs:
        try:
            potential_dir.mkdir(parents=True, exist_ok=True)
            
            # Testar se consegue escrever
            test_file = potential_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            
            # Se chegou até aqui, o diretório funciona
            log_dir = potential_dir
            log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = log_dir / log_filename
            break
            
        except (PermissionError, OSError) as e:
            continue
    
    if log_filepath is None:
        raise Exception("Não foi possível criar diretório de logs em nenhuma localização")
    
    # Configurar formato estruturado do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Limpar handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configurar logging com múltiplos handlers
    handlers = [
        logging.FileHandler(log_filepath, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # Console output
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger("IOT_Pipeline")
    
    # Log de inicialização
    if log_dir != possible_log_dirs[0]:
        logger.warning(f"Usando diretório alternativo para logs: {log_dir}")
    
    logger.info(f"Sistema de logging configurado - Arquivo: {log_filepath}")
    logger.info(f"Nível de log: {log_level}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtém um logger configurado
    
    Args:
        name: Nome do logger (opcional)
        
    Returns:
        Logger configurado
    """
    if name:
        return logging.getLogger(f"IOT_Pipeline.{name}")
    return logging.getLogger("IOT_Pipeline")
