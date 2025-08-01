"""
Módulo para coleta de dados meteorológicos da API OpenWeatherMap
"""
import requests
import json
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import api_key, cities, MONGO_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherFetcher:
    """Classe para buscar dados meteorológicos da API"""
    
    def __init__(self):
        self.api_key = api_key
        self.cities = cities
        self.mongo_config = MONGO_CONFIG
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_weather_data(self, lat: str, lon: str) -> Optional[Dict]:
        """
        Busca dados meteorológicos para uma localização específica
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict com dados meteorológicos ou None se erro
        """
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'  # Para temperatura em Celsius
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Adicionar timestamp de coleta
            weather_data['fetch_timestamp'] = datetime.now(timezone.utc).isoformat()
            weather_data['processed'] = False
            
            logger.debug(f"Dados coletados para lat={lat}, lon={lon}")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados para lat={lat}, lon={lon}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON para lat={lat}, lon={lon}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado para lat={lat}, lon={lon}: {e}")
            return None
    
    def save_to_mongodb(self, weather_data: Dict) -> bool:
        """
        Salva dados meteorológicos no MongoDB
        
        Args:
            weather_data: Dados meteorológicos
            
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
            
            # Inserir documento
            result = collection.insert_one(weather_data)
            
            logger.debug(f"Dados salvos no MongoDB: {result.inserted_id}")
            client.close()
            return True
            
        except PyMongoError as e:
            logger.error(f"Erro ao salvar no MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado ao salvar no MongoDB: {e}")
            return False
    
    def fetch_all_cities(self) -> Dict[str, int]:
        """
        Busca dados meteorológicos para todas as cidades configuradas
        
        Returns:
            Dict com estatísticas da coleta
        """
        logger.info(f"Iniciando coleta de dados para {len(self.cities)} cidades")
        
        stats = {
            'total_cities': len(self.cities),
            'successful_fetches': 0,
            'failed_fetches': 0,
            'successful_saves': 0,
            'failed_saves': 0
        }
        
        for city_info in self.cities:
            try:
                lat = city_info['lat']
                lon = city_info['lon']
                city_name = city_info.get('city', 'Unknown')
                
                # Buscar dados meteorológicos
                weather_data = self.get_weather_data(lat, lon)
                
                if weather_data:
                    stats['successful_fetches'] += 1
                    
                    # Adicionar informações da cidade
                    weather_data['city_info'] = city_info
                    
                    # Salvar no MongoDB
                    if self.save_to_mongodb(weather_data):
                        stats['successful_saves'] += 1
                        logger.debug(f"Dados salvos com sucesso para {city_name}")
                    else:
                        stats['failed_saves'] += 1
                        logger.warning(f"Falha ao salvar dados para {city_name}")
                else:
                    stats['failed_fetches'] += 1
                    logger.warning(f"Falha ao buscar dados para {city_name}")
                    
            except Exception as e:
                stats['failed_fetches'] += 1
                logger.error(f"Erro ao processar cidade {city_info}: {e}")
        
        # Log das estatísticas
        logger.info("Coleta finalizada:")
        logger.info(f"  Total de cidades: {stats['total_cities']}")
        logger.info(f"  Buscas bem-sucedidas: {stats['successful_fetches']}")
        logger.info(f"  Buscas falharam: {stats['failed_fetches']}")
        logger.info(f"  Salvamentos bem-sucedidos: {stats['successful_saves']}")
        logger.info(f"  Salvamentos falharam: {stats['failed_saves']}")
        
        return stats


def main():
    """Função principal para execução standalone"""
    try:
        logger.info("Iniciando coleta de dados meteorológicos")
        
        fetcher = WeatherFetcher()
        stats = fetcher.fetch_all_cities()
        
        if stats['successful_saves'] > 0:
            logger.info("Coleta executada com sucesso!")
        else:
            logger.error("Nenhum dado foi salvo com sucesso!")
            
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        raise


if __name__ == "__main__":
    main()
