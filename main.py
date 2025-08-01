"""
Arquivo principal para executar o pipeline IoT localmente
"""
import sys
import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logging
from src.weather_fetch import WeatherFetcher
from src.weather_hour import WeatherHourlyAggregator
from src.data_processing.weather_processor import WeatherDataProcessor
from src.ml.city_predictor import CityTemperaturePredictor


class IoTPipeline:
    """Classe principal para orquestrar o pipeline IoT"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.stats = {
            'start_time': datetime.now(),
            'extraction': {},
            'aggregation': {},
            'processing': {},
            'ml_training': {},
            'predictions': {},
            'total_duration': None,
            'success': False
        }
    
    def extract_weather_data(self) -> Dict[str, Any]:
        """Etapa 1: Extração de dados meteorológicos"""
        self.logger.info("="*60)
        self.logger.info("ETAPA 1: EXTRAÇÃO DE DADOS METEOROLÓGICOS")
        self.logger.info("="*60)
        
        try:
            fetcher = WeatherFetcher()
            stats = fetcher.fetch_all_cities()
            
            self.stats['extraction'] = stats
            self.logger.info(f"Extração concluída: {stats['successful_saves']} registros salvos")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro na extração de dados: {e}")
            raise
    
    def aggregate_hourly_data(self) -> Dict[str, Any]:
        """Etapa 2: Agregação horária de dados"""
        self.logger.info("="*60)
        self.logger.info("ETAPA 2: AGREGAÇÃO HORÁRIA DE DADOS")
        self.logger.info("="*60)
        
        try:
            aggregator = WeatherHourlyAggregator()
            stats = aggregator.process_hourly_aggregation(hours_back=1)
            aggregator.close()
            
            self.stats['aggregation'] = stats
            
            if stats['success']:
                self.logger.info(f"Agregação concluída: {stats['records_saved']} registros processados")
            else:
                raise Exception(f"Falha na agregação: {stats.get('error_message', 'Erro desconhecido')}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro na agregação de dados: {e}")
            raise
    
    def process_data_for_ml(self) -> Dict[str, Any]:
        """Etapa 3: Processamento de dados para ML"""
        self.logger.info("="*60)
        self.logger.info("ETAPA 3: PROCESSAMENTO DE DADOS PARA ML")
        self.logger.info("="*60)
        
        try:
            processor = WeatherDataProcessor()
            
            # Carregar dados processados
            df_raw = processor.load_processed_data(source='iot_weather_db')
            self.logger.info(f"Dados carregados: {len(df_raw)} registros")
            
            # Limpar dados
            df_clean = processor.clean_data(df_raw)
            self.logger.info(f"Dados limpos: {len(df_clean)} registros")
            
            # Feature engineering
            df_processed = processor.feature_engineering(df_clean)
            self.logger.info(f"Features criadas: {len(df_processed)} registros")
            
            stats = {
                'raw_records': len(df_raw),
                'clean_records': len(df_clean),
                'processed_records': len(df_processed),
                'data_quality_score': self._calculate_data_quality(df_processed)
            }
            
            self.stats['processing'] = stats
            return {'df_processed': df_processed, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"Erro no processamento de dados: {e}")
            raise
    
    def train_ml_models(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Etapa 4: Treinamento de modelos de ML"""
        self.logger.info("="*60)
        self.logger.info("ETAPA 4: TREINAMENTO DE MODELOS DE ML")
        self.logger.info("="*60)
        
        try:
            # Filtrar dados recentes (últimos 30 dias)
            last_days = 30
            df_recent = df_processed[
                df_processed['date'] >= df_processed['date'].max() - pd.Timedelta(days=last_days)
            ].copy()
            self.logger.info(f"Últimos {last_days} dias: {len(df_recent)} registros")
            
            # Filtrar por algumas cidades específicas
            df_filtered = df_recent[
                (df_recent['idcity'] == 22) | 
                (df_recent['idcity'] == 29) | 
                (df_recent['idcity'] == 47)
            ]
            self.logger.info(f"Dados filtrados por cidades: {len(df_filtered)} registros")
            
            if len(df_filtered) < 100:
                self.logger.warning("Poucos dados disponíveis para treinamento")
                return {'status': 'skipped', 'reason': 'insufficient_data'}
            
            # Configurar preditor
            predictor = CityTemperaturePredictor(
                order=(2, 1, 1),
                seasonal_order=(1, 1, 1, 24),
                exog_lags=2
            )
            
            # Treinar modelos
            exog_columns = ['humidity', 'pressure', 'wind_speed', 'precipitation']
            trained_cities = predictor.train_city_models(
                df_filtered,
                target_column='temperature',
                city_column='idcity',
                exog_columns=exog_columns,
                test_size=0.2
            )
            self.logger.info(f"Modelos treinados para {len(trained_cities)} cidades")
            
            # Avaliar modelos
            metrics_by_city = predictor.evaluate_models()
            
            stats = {
                'trained_cities': len(trained_cities),
                'records_used': len(df_filtered),
                'cities_list': trained_cities,
                'metrics': metrics_by_city
            }
            
            self.stats['ml_training'] = stats
            return {'predictor': predictor, 'trained_cities': trained_cities, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento de modelos: {e}")
            raise
    
    def make_predictions(self, predictor: CityTemperaturePredictor, trained_cities: list) -> Dict[str, Any]:
        """Etapa 5: Fazer predições futuras"""
        self.logger.info("="*60)
        self.logger.info("ETAPA 5: GERANDO PREDIÇÕES FUTURAS")
        self.logger.info("="*60)
        
        try:
            predictions_summary = {}
            
            for city in trained_cities[:3]:  # Fazer predições para as primeiras 3 cidades
                try:
                    predictions, conf_intervals = predictor.predict_future(city, steps=24)
                    predictions_summary[city] = {
                        'next_24h_avg': float(predictions.mean()),
                        'min_predicted': float(predictions.min()),
                        'max_predicted': float(predictions.max()),
                        'predictions_count': len(predictions),
                        'std_deviation': float(predictions.std())
                    }
                    self.logger.info(f"Predições para cidade {city}: temp média próximas 24h = {predictions.mean():.2f}°C")
                except Exception as e:
                    self.logger.error(f"Erro ao fazer predições para cidade {city}: {e}")
            
            stats = {
                'cities_predicted': len(predictions_summary),
                'predictions': predictions_summary
            }
            
            self.stats['predictions'] = stats
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro na geração de predições: {e}")
            raise
    
    def generate_final_report(self):
        """Gerar relatório final do pipeline"""
        self.logger.info("="*60)
        self.logger.info("RELATÓRIO FINAL - PIPELINE IoT COMPLETO")
        self.logger.info("="*60)
        
        end_time = datetime.now()
        self.stats['total_duration'] = end_time - self.stats['start_time']
        
        # Resumo da execução
        self.logger.info(f"Início: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Duração total: {self.stats['total_duration']}")
        
        # Estatísticas de extração
        if self.stats['extraction']:
            ext = self.stats['extraction']
            self.logger.info(f"\nExtração de dados:")
            self.logger.info(f"  Cidades processadas: {ext['total_cities']}")
            self.logger.info(f"  Dados coletados: {ext['successful_fetches']}")
            self.logger.info(f"  Dados salvos: {ext['successful_saves']}")
        
        # Estatísticas de agregação
        if self.stats['aggregation']:
            agg = self.stats['aggregation']
            self.logger.info(f"\nAgregação horária:")
            self.logger.info(f"  Registros processados: {agg['records_processed']}")
            self.logger.info(f"  Registros salvos: {agg['records_saved']}")
            self.logger.info(f"  Status: {'Sucesso' if agg['success'] else 'Falha'}")
        
        # Estatísticas de processamento
        if self.stats['processing']:
            proc = self.stats['processing']
            self.logger.info(f"\nProcessamento de dados:")
            self.logger.info(f"  Registros brutos: {proc['raw_records']}")
            self.logger.info(f"  Registros limpos: {proc['clean_records']}")
            self.logger.info(f"  Registros processados: {proc['processed_records']}")
            self.logger.info(f"  Qualidade dos dados: {proc['data_quality_score']:.2%}")
        
        # Estatísticas de ML
        if self.stats['ml_training']:
            ml = self.stats['ml_training']
            self.logger.info(f"\nTreinamento de ML:")
            self.logger.info(f"  Cidades treinadas: {ml['trained_cities']}")
            self.logger.info(f"  Registros utilizados: {ml['records_used']}")
            
            # Métricas médias
            if ml['metrics']:
                avg_mae = sum(m['mae'] for m in ml['metrics'].values()) / len(ml['metrics'])
                avg_rmse = sum(m['rmse'] for m in ml['metrics'].values()) / len(ml['metrics'])
                avg_r2 = sum(m['r2'] for m in ml['metrics'].values()) / len(ml['metrics'])
                
                self.logger.info(f"  MAE médio: {avg_mae:.2f}°C")
                self.logger.info(f"  RMSE médio: {avg_rmse:.2f}°C")
                self.logger.info(f"  R² médio: {avg_r2:.4f}")
        
        # Estatísticas de predições
        if self.stats['predictions']:
            pred = self.stats['predictions']
            self.logger.info(f"\nPredições futuras:")
            self.logger.info(f"  Cidades com predições: {pred['cities_predicted']}")
            
            for city, prediction in pred['predictions'].items():
                self.logger.info(f"  Cidade {city}: {prediction['min_predicted']:.1f}°C - "
                              f"{prediction['max_predicted']:.1f}°C (média: {prediction['next_24h_avg']:.1f}°C)")
        
        self.logger.info("="*60)
        self.logger.info("PIPELINE IoT EXECUTADO COM SUCESSO!")
        self.logger.info("="*60)
        
        self.stats['success'] = True
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calcula uma pontuação de qualidade dos dados"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_score = 1 - (missing_cells / total_cells)
            return max(0.0, min(1.0, quality_score))
        except:
            return 0.0
    
    def run(self):
        """Executa o pipeline completo"""
        try:
            self.logger.info("INICIANDO PIPELINE IoT COMPLETO")
            
            # Etapa 1: Extração
            self.extract_weather_data()
            
            # Etapa 2: Agregação
            self.aggregate_hourly_data()
            
            # Etapa 3: Processamento
            processing_result = self.process_data_for_ml()
            df_processed = processing_result['df_processed']
            
            # Etapa 4: Treinamento ML
            ml_result = self.train_ml_models(df_processed)
            
            if ml_result.get('status') != 'skipped':
                predictor = ml_result['predictor']
                trained_cities = ml_result['trained_cities']
                
                # Etapa 5: Predições
                self.make_predictions(predictor, trained_cities)
            
            # Relatório final
            self.generate_final_report()
            
        except Exception as e:
            self.logger.error(f"Erro na execução do pipeline: {e}")
            self.stats['success'] = False
            raise


def main():
    """Função principal para executar o pipeline"""
    try:
        pipeline = IoTPipeline()
        pipeline.run()
        
    except Exception as e:
        print(f"Erro fatal no pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
