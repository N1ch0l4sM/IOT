"""
Utilitários para conexão com banco de dados
"""
import logging
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config import DB_CONFIG

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Classe para gerenciar conexões com o banco de dados"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """Estabelece conexão com o banco de dados"""
        try:
            connection_string = (
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )
            
            self.engine = create_engine(connection_string, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info("Conexão com banco de dados estabelecida")
            
        except Exception as e:
            logger.error(f"Erro ao conectar com banco de dados: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Executa query e retorna DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params=params)
                return result
                
        except Exception as e:
            logger.error(f"Erro ao executar query: {e}")
            raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Insere DataFrame no banco de dados"""
        try:
            # Usar o engine diretamente com pandas
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, method='multi')
            logger.info(f"Dados inseridos na tabela {table_name}")
            
        except Exception as e:
            logger.error(f"Erro ao inserir dados: {e}")
            raise
    
    def get_session(self):
        """Retorna sessão do banco de dados"""
        return self.Session()
    
    def close(self):
        """Fecha conexão com banco de dados"""
        if self.engine:
            self.engine.dispose()
            logger.info("Conexão com banco de dados fechada")

# Instância global
db_connection = DatabaseConnection()
