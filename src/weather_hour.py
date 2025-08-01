"""
Módulo para agregação de dados meteorológicos por hora
Processa dados do MongoDB e salva agregações no PostgreSQL
"""
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round as spark_round
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import MONGO_CONFIG, PG2_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherHourlyAggregator:
    """Classe para agregação horária de dados meteorológicos"""
    
    def __init__(self):
        self.mongo_config = MONGO_CONFIG
        self.postgres_config = PG2_CONFIG
        self.spark = None
        self._init_spark_session()
        
    def _init_spark_session(self):
        """Inicializa sessão Spark"""
        try:
            # Configurar Spark com MongoDB usando opções individuais
            self.spark = (
                SparkSession.builder
                .appName("WeatherHourlyAggregation")
                .config("spark.jars.packages", 
                       "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0,"
                       "org.postgresql:postgresql:42.6.0")
                .config("spark.mongodb.input.host", f"{self.mongo_config['host']}:{self.mongo_config['port']}")
                .config("spark.mongodb.input.database", self.mongo_config['db'])
                .config("spark.mongodb.input.collection", self.mongo_config['collection'])
                .config("spark.mongodb.input.username", self.mongo_config['username'])
                .config("spark.mongodb.input.password", self.mongo_config['password'])
                .config("spark.mongodb.input.authSource", "admin")
                .config("spark.sql.adaptive.enabled", "false")
                .getOrCreate()
            )
            
            # Reduzir verbosidade dos logs do Spark
            self.spark.sparkContext.setLogLevel("WARN")
            
            logger.info("SparkSession iniciada para agregação horária")
        except Exception as e:
            logger.error(f"Erro ao iniciar SparkSession: {e}")
            raise
    
    def get_postgres_properties(self) -> Dict[str, str]:
        """Retorna propriedades de conexão PostgreSQL"""
        return {
            "user": self.postgres_config['user'],
            "password": self.postgres_config['password'],
            "driver": "org.postgresql.Driver"
        }
    
    def get_postgres_url(self) -> str:
        """Retorna URL de conexão PostgreSQL"""
        return (f"jdbc:postgresql://{self.postgres_config['host']}:"
                f"{self.postgres_config['port']}/{self.postgres_config['database']}")
    
    def test_connections(self) -> bool:
        """
        Testa as conexões com MongoDB e PostgreSQL
        
        Returns:
            True se ambas as conexões funcionam, False caso contrário
        """
        logger.info("Testando conexões...")
        
        # Testar MongoDB com PyMongo
        try:
            client = MongoClient(
                host=self.mongo_config['host'],
                port=self.mongo_config['port'],
                username=self.mongo_config['username'],
                password=self.mongo_config['password'],
                authSource='admin'
            )
            
            # Testar conexão
            client.admin.command('ping')
            
            # Verificar se a collection existe e tem dados
            db = client[self.mongo_config['db']]
            if self.mongo_config['collection'] not in db.list_collection_names():
                logger.error(f"Collection '{self.mongo_config['collection']}' não encontrada")
                client.close()
                return False
                
            collection = db[self.mongo_config['collection']]
            count = collection.count_documents({})
            logger.info(f"✅ MongoDB conectado. Documents na collection: {count}")
            
            if count == 0:
                logger.warning("⚠️  Collection está vazia")
            
            client.close()
            
        except Exception as e:
            logger.error(f"❌ Erro na conexão MongoDB: {e}")
            return False
        
        # Testar PostgreSQL com Spark
        try:
            test_df = (
                self.spark.read
                .jdbc(
                    url=self.get_postgres_url(),
                    table="city",
                    properties=self.get_postgres_properties()
                )
            )
            city_count = test_df.count()
            logger.info(f"✅ PostgreSQL conectado. Cidades carregadas: {city_count}")
            
        except Exception as e:
            logger.error(f"❌ Erro na conexão PostgreSQL: {e}")
            return False
        
        return True
    
    def load_weather_data(self, hours_back: int = 1) -> Optional[any]:
        """
        Carrega dados meteorológicos do MongoDB para o período especificado
        
        Args:
            hours_back: Quantas horas atrás buscar dados
            
        Returns:
            Spark DataFrame ou None se erro
        """
        try:
            # Calcular timestamp limite
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            timestamp_limit = time_limit.timestamp()
            
            logger.info(f"Carregando dados MongoDB desde: {time_limit} (timestamp: {timestamp_limit})")
            logger.info(f"MongoDB config: host={self.mongo_config['host']}, db={self.mongo_config['db']}, collection={self.mongo_config['collection']}")
            
            # Carregar dados do MongoDB com configurações explícitas
            weather_df = (
                self.spark.read
                .format("mongo")
                .option("host", f"{self.mongo_config['host']}:{self.mongo_config['port']}")
                .option("database", self.mongo_config['db'])
                .option("collection", self.mongo_config['collection'])
                .option("username", self.mongo_config['username'])
                .option("password", self.mongo_config['password'])
                .option("authSource", "admin")
                .load()
            )
            
            # Verificar se dados foram carregados
            total_count = weather_df.count()
            logger.info(f"Total de documentos carregados: {total_count}")
            
            if total_count == 0:
                logger.warning("Nenhum documento encontrado no MongoDB")
                return None
            
            # Filtrar dados recentes
            filtered_df = weather_df.filter(col("dt") >= timestamp_limit)
            filtered_count = filtered_df.count()
            
            logger.info(f"Dados filtrados das últimas {hours_back} hora(s): {filtered_count} registros")
            
            if filtered_count == 0:
                logger.warning(f"Nenhum dado encontrado para as últimas {hours_back} hora(s)")
                return None
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados do MongoDB: {e}")
            logger.error(f"Detalhes do erro: {str(e)}")
            return None
    
    def aggregate_weather_data(self, weather_df) -> Optional[any]:
        """
        Agrega dados meteorológicos por coordenadas
        
        Args:
            weather_df: DataFrame com dados meteorológicos
            
        Returns:
            DataFrame agregado ou None se erro
        """
        try:
            # Criar view temporária
            weather_df.createOrReplaceTempView("weather_raw")
            
            # Query SQL para agregação
            aggregation_sql = """
            SELECT 
                ROUND(coord.lon, 2) AS lon, 
                ROUND(coord.lat, 2) AS lat,
                ROUND(AVG(main.temp), 2) AS avg_temperature,
                ROUND(AVG(main.humidity), 2) AS avg_humidity,
                ROUND(AVG(COALESCE(rain.`1h`, 0)), 2) AS avg_precipitation,
                ROUND(AVG(wind.speed), 2) AS avg_wind_speed,
                ROUND(AVG(clouds.all), 2) AS avg_clouds,
                ROUND(AVG(main.pressure), 2) AS avg_pressure,
                ROUND(AVG(main.feels_like), 2) AS avg_feels_like,
                COUNT(*) as measurement_count,
                MAX(dt) as latest_measurement
            FROM weather_raw
            GROUP BY ROUND(coord.lon, 2), ROUND(coord.lat, 2)
            """
            
            aggregated_df = self.spark.sql(aggregation_sql)
            logger.info("Dados agregados por coordenadas")
            
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Erro na agregação de dados: {e}")
            return None
    
    def load_city_data(self) -> Optional[any]:
        """
        Carrega dados das cidades do PostgreSQL
        
        Returns:
            DataFrame com dados das cidades ou None se erro
        """
        try:
            city_df = (
                self.spark.read
                .jdbc(
                    url=self.get_postgres_url(),
                    table="city",
                    properties=self.get_postgres_properties()
                )
            )
            logger.info("Dados das cidades carregados do PostgreSQL")
            return city_df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados das cidades: {e}")
            return None
    
    def join_weather_with_cities(self, weather_df, city_df) -> Optional[any]:
        """
        Faz join dos dados meteorológicos com as cidades
        
        Args:
            weather_df: DataFrame com dados meteorológicos agregados
            city_df: DataFrame com dados das cidades
            
        Returns:
            DataFrame com join ou None se erro
        """
        try:
            # Criar views temporárias
            weather_df.createOrReplaceTempView("weather_aggregated")
            city_df.createOrReplaceTempView("cities")
            
            # Query SQL para join
            join_sql = """
            SELECT 
                c.idCity AS city_id,
                c.CityName AS city_name,
                c.Country AS country,
                w.avg_temperature,
                w.avg_humidity,
                w.avg_precipitation,
                w.avg_wind_speed,
                w.avg_clouds,
                w.avg_feels_like,
                w.avg_pressure,
                w.measurement_count,
                w.latest_measurement,
                w.lon,
                w.lat
            FROM weather_aggregated w
            JOIN cities c 
                ON ROUND(w.lat, 2) = ROUND(c.lat, 2)
                AND ROUND(w.lon, 2) = ROUND(c.lon, 2)
            """
            
            joined_df = self.spark.sql(join_sql)
            logger.info("Join realizado entre dados meteorológicos e cidades")
            
            return joined_df
            
        except Exception as e:
            logger.error(f"Erro no join de dados: {e}")
            return None
    
    def prepare_hourly_data(self, joined_df) -> Optional[any]:
        """
        Prepara dados para inserção na tabela weather_hour
        
        Args:
            joined_df: DataFrame com dados unidos
            
        Returns:
            DataFrame preparado ou None se erro
        """
        try:
            # Calcular data e hora atuais
            current_time = datetime.now() - timedelta(hours=1)
            current_date = current_time.strftime("%Y-%m-%d")
            current_hour = current_time.hour
            
            # Criar view temporária
            joined_df.createOrReplaceTempView("joined_data")
            
            # Query SQL para preparar dados finais
            final_sql = f"""
            SELECT 
                city_id AS idCity,
                CAST('{current_date}' AS date) AS Date,
                {current_hour} AS Hour,
                avg_temperature AS Temp,
                avg_feels_like AS FeelsLike,
                avg_clouds AS Clouds,
                avg_precipitation AS Rain,
                avg_wind_speed AS Wind,
                avg_pressure AS Pressure,
                avg_humidity AS Humidity,
                measurement_count,
                CURRENT_TIMESTAMP() AS created_at
            FROM joined_data
            """
            
            final_df = self.spark.sql(final_sql)
            logger.info("Dados preparados para inserção")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados finais: {e}")
            return None
    
    def save_to_postgres(self, df) -> bool:
        """
        Salva dados agregados no PostgreSQL
        
        Args:
            df: DataFrame para salvar
            
        Returns:
            True se sucesso, False se erro
        """
        try:
            df.write.mode("append").jdbc(
                url=self.get_postgres_url(),
                table="weather_hour",
                properties=self.get_postgres_properties()
            )
            
            logger.info("Dados salvos na tabela weather_hour")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados no PostgreSQL: {e}")
            return False
    
    def mark_data_as_processed(self, hours_back: int = 1) -> bool:
        """
        Marca dados no MongoDB como processados
        
        Args:
            hours_back: Quantas horas atrás marcar como processados
            
        Returns:
            True se sucesso, False se erro
        """
        try:
            client = MongoClient(
                host=self.mongo_config['host'],
                port=self.mongo_config['port'],
                username=self.mongo_config['username'],
                password=self.mongo_config['password'],
                authSource='admin'
            )
            
            db = client[self.mongo_config['db']]
            collection = db[self.mongo_config['collection']]
            
            # Calcular timestamp limite
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            timestamp_limit = time_limit.timestamp()
            
            # Atualizar documentos
            result = collection.update_many(
                {"dt": {"$gte": timestamp_limit}, "processed": {"$ne": True}},
                {"$set": {"processed": True, "processed_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"Marcados {result.modified_count} documentos como processados")
            client.close()
            return True
            
        except PyMongoError as e:
            logger.error(f"Erro ao marcar dados como processados: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado ao marcar dados: {e}")
            return False
    
    def process_hourly_aggregation(self, hours_back: int = 1) -> Dict[str, any]:
        """
        Executa o processo completo de agregação horária
        
        Args:
            hours_back: Quantas horas atrás processar
            
        Returns:
            Dict com estatísticas do processamento
        """
        logger.info(f"Iniciando agregação horária para últimas {hours_back} hora(s)")
        
        stats = {
            'start_time': datetime.now(),
            'success': False,
            'records_processed': 0,
            'records_saved': 0,
            'error_message': None
        }
        
        try:
            # 0. Testar conexões primeiro
            if not self.test_connections():
                raise Exception("Falha nos testes de conexão")
            
            # 1. Carregar dados meteorológicos
            weather_df = self.load_weather_data(hours_back)
            if weather_df is None:
                raise Exception("Falha ao carregar dados meteorológicos")
            
            # 2. Agregar dados
            aggregated_df = self.aggregate_weather_data(weather_df)
            if aggregated_df is None:
                raise Exception("Falha na agregação de dados")
            
            stats['records_processed'] = aggregated_df.count()
            
            # 3. Carregar dados das cidades
            city_df = self.load_city_data()
            if city_df is None:
                raise Exception("Falha ao carregar dados das cidades")
            
            # 4. Fazer join
            joined_df = self.join_weather_with_cities(aggregated_df, city_df)
            if joined_df is None:
                raise Exception("Falha no join de dados")
            
            # 5. Preparar dados finais
            final_df = self.prepare_hourly_data(joined_df)
            if final_df is None:
                raise Exception("Falha na preparação dos dados finais")
            
            stats['records_saved'] = final_df.count()
            
            # 6. Salvar no PostgreSQL
            if not self.save_to_postgres(final_df):
                raise Exception("Falha ao salvar dados no PostgreSQL")
            
            # 7. Marcar dados como processados
            self.mark_data_as_processed(hours_back)
            
            stats['success'] = True
            stats['end_time'] = datetime.now()
            stats['duration'] = stats['end_time'] - stats['start_time']
            
            logger.info("Agregação horária concluída com sucesso")
            logger.info(f"Registros processados: {stats['records_processed']}")
            logger.info(f"Registros salvos: {stats['records_saved']}")
            logger.info(f"Duração: {stats['duration']}")
            
        except Exception as e:
            stats['error_message'] = str(e)
            stats['end_time'] = datetime.now()
            logger.error(f"Erro na agregação horária: {e}")
        
        return stats
    
    def close(self):
        """Fecha SparkSession"""
        if self.spark:
            self.spark.stop()
            logger.info("SparkSession encerrada")


def main():
    """Função principal para execução standalone"""
    try:
        logger.info("Iniciando agregação horária de dados meteorológicos")
        
        aggregator = WeatherHourlyAggregator()
        stats = aggregator.process_hourly_aggregation(hours_back=1)
        
        if stats['success']:
            logger.info("Agregação executada com sucesso!")
        else:
            logger.error(f"Falha na agregação: {stats['error_message']}")
            
        aggregator.close()
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        raise


if __name__ == "__main__":
    main()