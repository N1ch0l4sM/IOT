"""
aaaaUtilitários para conexão com banco de dados usando Spark e SparkSQL
"""
import logging
from typing import Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.utils import AnalysisException
from src.config import DB_CONFIG

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Classe para gerenciar conexões com o banco de dados via Spark"""
    
    def __init__(self):
        self.spark = None
        self._connect()
    
    def _connect(self):
        """Estabelece conexão Spark"""
        try:
            self.spark = (
                SparkSession.builder
                .appName("IOT-DB-Connection")
                .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0")
                .getOrCreate()
            )
            logger.info("SparkSession iniciada para conexão com banco de dados")
        except Exception as e:
            logger.error(f"Erro ao iniciar SparkSession: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Executa query SQL no banco de dados Postgres via Spark e retorna DataFrame Pandas.
        """
        try:
            jdbc_url = (
                f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )
            properties = {
                "user": DB_CONFIG['user'],
                "password": DB_CONFIG['password'],
                "driver": "org.postgresql.Driver"
            }
            # Substituição simples de parâmetros (atenção: use apenas para queries seguras)
            if params:
                for k, v in params.items():
                    query = query.replace(f":{k}", str(v))
            df = self.spark.read.jdbc(
                url=jdbc_url,
                table=f"({query}) as subquery",
                properties=properties
            )
            return df.toPandas()
        except AnalysisException as e:
            logger.error(f"Erro de análise SparkSQL: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro ao executar query via Spark: {e}")
            raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        Insere DataFrame Pandas no banco de dados Postgres via Spark.
        """
        try:
            # Mantenha apenas as colunas da tabela
            expected_columns = [
                "location", "temperature", "humidity", "pressure", "wind_speed",
                "wind_direction", "precipitation", "rain_probability", "recorded_at"
            ]
            df = df[expected_columns]

            # Ajuste de tipos
            df = df.astype({
                "location": "str",
                "temperature": "float",
                "humidity": "float",
                "pressure": "float",
                "wind_speed": "float",
                "wind_direction": "str",
                "precipitation": "float",
                "rain_probability": "float",
                "recorded_at": "datetime64[ns]"
            })
            spark_df = self.spark.createDataFrame(df)
            jdbc_url = (
                f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )
            mode = 'append' if if_exists == 'append' else 'overwrite'
            spark_df.write.jdbc(
                url=jdbc_url,
                table=table_name,
                mode=mode,
                properties={
                    "user": DB_CONFIG['user'],
                    "password": DB_CONFIG['password'],
                    "driver": "org.postgresql.Driver"
                }
            )
            logger.info(f"Dados inseridos na tabela {table_name} via Spark")
        except Exception as e:
            logger.error(f"Erro ao inserir dados via Spark: {e}")
            raise
    
    def get_session(self):
        """Retorna SparkSession"""
        return self.spark
    
    def close(self):
        """Fecha SparkSession"""
        if self.spark:
            self.spark.stop()
            logger.info("SparkSession encerrada")

# Instância global
db_connection = DatabaseConnection()
