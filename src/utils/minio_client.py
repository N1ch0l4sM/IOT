"""
Utilitários para conexão com MinIO
"""
import logging
from typing import Optional
import pandas as pd
from minio import Minio
from minio.error import S3Error
import io
from src.config import MINIO_CONFIG

logger = logging.getLogger(__name__)

class MinIOConnection:
    """Classe para gerenciar conexões com MinIO"""
    
    def __init__(self):
        self.client = None
        self.bucket_name = MINIO_CONFIG['bucket']
    
    def _ensure_connected(self):
        """Garante que existe uma conexão ativa com MinIO"""
        if self.client is None:
            self._connect()
    
    def _connect(self):
        """Estabelece conexão com MinIO"""
        try:
            self.client = Minio(
                MINIO_CONFIG['endpoint'],
                access_key=MINIO_CONFIG['access_key'],
                secret_key=MINIO_CONFIG['secret_key'],
                secure=False
            )
            
            # Criar bucket se não existir
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} criado")
            
            logger.info("Conexão com MinIO estabelecida")
            
        except Exception as e:
            logger.error(f"Erro ao conectar com MinIO: {e}")
            raise
    
    def upload_dataframe(self, df: pd.DataFrame, object_name: str, format: str = 'csv'):
        """Faz upload de DataFrame para MinIO"""
        try:
            self._ensure_connected()
            
            # Converter DataFrame para bytes
            if format == 'csv':
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                data = buffer.getvalue().encode('utf-8')
            elif format == 'parquet':
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                data = buffer.getvalue()
            else:
                raise ValueError(f"Formato {format} não suportado")
            
            # Upload para MinIO
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(data),
                len(data),
                content_type='application/octet-stream'
            )
            
            logger.info(f"DataFrame uploaded para {object_name}")
            
        except S3Error as e:
            logger.error(f"Erro ao fazer upload: {e}")
            raise
    
    def download_dataframe(self, object_name: str, format: str = 'csv') -> pd.DataFrame:
        """Faz download de DataFrame do MinIO"""
        try:
            self._ensure_connected()
            
            # Download do MinIO
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            
            # Converter bytes para DataFrame
            if format == 'csv':
                df = pd.read_csv(io.StringIO(data.decode('utf-8')))
            elif format == 'parquet':
                df = pd.read_parquet(io.BytesIO(data))
            else:
                raise ValueError(f"Formato {format} não suportado")
            
            logger.info(f"DataFrame baixado de {object_name}")
            return df
            
        except S3Error as e:
            logger.error(f"Erro ao fazer download: {e}")
            raise
    
    def list_objects(self, prefix: str = '') -> list:
        """Lista objetos no bucket"""
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
            
        except S3Error as e:
            logger.error(f"Erro ao listar objetos: {e}")
            raise
    
    def delete_object(self, object_name: str):
        """Deleta objeto do MinIO"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Objeto {object_name} deletado")
            
        except S3Error as e:
            logger.error(f"Erro ao deletar objeto: {e}")
            raise

# Instância global
minio_connection = MinIOConnection()
