"""
Processamento e limpeza de dados climáticos
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from src.utils.database import db_connection
from src.utils.minio_client import minio_connection
import kagglehub 

logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    """Classe para processamento de dados climáticos"""
    
    def __init__(self):
        self.required_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 
            'wind_direction', 'precipitation'
        ]
    
    def load_raw_data(self, source: str = 'kaggle') -> pd.DataFrame:
        """Carrega dados brutos de diferentes fontes"""
        try:
            if source == 'kaggle':
                path = kagglehub.dataset_download("nelgiriyewithana/global-weather-repository")
                df = pd.read_csv(path + '/GlobalWeatherRepository.csv')
                logger.info(f"Dados carregados do Kaggle: {len(df)} registros")
                # Limpeza inicial dos dados
                df = self.clean_data_kaggle(df)
                return df
            
            elif source == 'minio':
                # Carregar dados do MinIO
                df = minio_connection.download_dataframe('raw/weather_data.csv')
                logger.info(f"Dados carregados do MinIO: {len(df)} registros")
                return df
            
            else:
                raise ValueError(f"Fonte de dados '{source}' não suportada")
                
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def clean_data_kaggle(self, df):
        # Drop column: 'pressure_in'
        df = df.drop(columns=['pressure_in'])
        # Drop column: 'temperature_fahrenheit'
        df = df.drop(columns=['temperature_fahrenheit'])
        # Drop column: 'wind_mph'
        df = df.drop(columns=['wind_mph'])
        # Drop column: 'precip_in'
        df = df.drop(columns=['precip_in'])
        # Drop column: 'feels_like_fahrenheit'
        df = df.drop(columns=['feels_like_fahrenheit'])
        # Drop column: 'visibility_miles'
        df = df.drop(columns=['visibility_miles'])
        # Drop column: 'gust_mph'
        df = df.drop(columns=['gust_mph'])
        # Rename column 'temperature_celsius' to 'temperature'
        df = df.rename(columns={'temperature_celsius': 'temperature'})
        # Rename column 'pressure_mb' to 'pressure'
        df = df.rename(columns={'pressure_mb': 'pressure'})
        # Rename column 'wind_kph' to 'wind_speed'
        df = df.rename(columns={'wind_kph': 'wind_speed'})
        # Rename column 'precip_mm' to 'precipitation'
        df = df.rename(columns={'precip_mm': 'precipitation'})
        # Rename column 'last_updated' to 'recorded_at'
        df = df.rename(columns={'last_updated': 'recorded_at'})
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida dados"""
        try:
            logger.info("Iniciando limpeza de dados")
            
            # Remover duplicatas
            df = df.drop_duplicates()
            
            # Tratar valores ausentes
            df = self._handle_missing_values(df)
            
            # Remover outliers
            df = self._remove_outliers(df)
            
            # Validar tipos de dados
            df = self._validate_data_types(df)
            
            logger.info(f"Limpeza concluída: {len(df)} registros restantes")
            return df
            
        except Exception as e:
            logger.error(f"Erro na limpeza de dados: {e}")
            raise
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features adicionais"""
        try:
            logger.info("Iniciando feature engineering")
            
            # Criar features temporais
            if 'recorded_at' in df.columns:
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])
                df['hour'] = df['recorded_at'].dt.hour
                df['day_of_week'] = df['recorded_at'].dt.dayofweek
                df['month'] = df['recorded_at'].dt.month
                df['season'] = df['month'].apply(self._get_season)
            
            # Criar features meteorológicas
            df['feels_like'] = self._calculate_feels_like(df['temperature'], df['humidity'])
            df['dew_point'] = self._calculate_dew_point(df['temperature'], df['humidity'])
            df['pressure_tendency'] = df['pressure'].diff().fillna(0)
            
            # Criar features de interação
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
            
            # Criar target (chuva ou não)
            df['will_rain'] = (df['precipitation'] > 0).astype(int)
            
            logger.info("Feature engineering concluído")
            return df
            
        except Exception as e:
            logger.error(f"Erro no feature engineering: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, destination: str = 'both'):
        """Salva dados processados"""
        try:
            if destination in ['database', 'both']:
                # Salvar no banco de dados
                db_connection.insert_dataframe(df, 'weather_data', if_exists='replace')
                logger.info("Dados salvos no banco de dados")
            
            if destination in ['minio', 'both']:
                # Salvar no MinIO
                minio_connection.upload_dataframe(
                    df, 'processed/weather_data_processed.csv', format='csv'
                )
                logger.info("Dados salvos no MinIO")
                
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores ausentes"""
        # Preencher valores numéricos com mediana
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Preencher valores categóricos com moda
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers usando IQR"""
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e converte tipos de dados"""
        # Garantir que valores numéricos são float
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Garantir que umidade está entre 0 e 100
        if 'humidity' in df.columns:
            df['humidity'] = df['humidity'].clip(0, 100)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """Retorna estação do ano baseada no mês"""
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    def _calculate_feels_like(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calcula sensação térmica"""
        # Fórmula simplificada de heat index
        return temp + 0.5 * (humidity / 100) * (temp - 14.5)
    
    def _calculate_dew_point(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calcula ponto de orvalho"""
        # Fórmula de Magnus
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100)
        return (b * alpha) / (a - alpha)
