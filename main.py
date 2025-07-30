"""
Arquivo principal para executar o pipeline localmente
"""
import sys
import os
from datetime import datetime

import pandas as pd

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logging
from src.data_processing.weather_processor import WeatherDataProcessor
from src.ml.city_predictor import CityTemperaturePredictor

def main():
    """Função principal para executar o pipeline"""
    
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando pipeline de previsão de temperatura por cidade")
    
    try:
        # Etapa 1: Processamento de dados
        logger.info("Etapa 1: Processamento de dados")
        processor = WeatherDataProcessor()
        
        # Carregar dados processados
        df_raw = processor.load_processed_data(source='iot_weather_db')
        logger.info(f"Dados processados carregados: {len(df_raw)} registros")
        
        # Limpar dados
        df_clean = processor.clean_data(df_raw)
        logger.info(f"Dados limpos: {len(df_clean)} registros")
        
        # Feature engineering
        df_processed = processor.feature_engineering(df_clean)
        logger.info(f"Features criadas: {len(df_processed)} registros")
        
        # Etapa 2: Treinamento do modelo de temperatura por cidade
        logger.info("Etapa 2: Treinamento do modelo de temperatura por cidade")
        predictor = CityTemperaturePredictor(
            order=(2, 1, 1), 
            seasonal_order=(1, 1, 1, 24),
            exog_lags=2
        )
        
        # Definir variáveis exógenas para o modelo
        exog_columns = ['humidity', 'pressure', 'wind_speed', 'precipitation']
        last_days = 30
        df_recent = df_processed[df_processed['date'] >= df_processed['date'].max() - pd.Timedelta(days=last_days)].copy()
        print(f"Últimos {last_days} dias: {len(df_recent)} registros")

        df_processed = df_recent[(df_recent['idcity'] == 22) | (df_recent['idcity'] == 29) | (df_recent['idcity'] == 47)]  # Filtrar por uma cidade específica para exemplo

        # Treinar modelos por cidade
        trained_cities = predictor.train_city_models(
            df_processed,
            target_column='temperature',
            city_column='idcity',
            exog_columns=exog_columns,
            test_size=0.2
        )
        logger.info(f"Modelos treinados para {len(trained_cities)} cidades")
        
        # Etapa 3: Avaliação dos modelos
        logger.info("Etapa 3: Avaliação dos modelos")
        metrics_by_city = predictor.evaluate_models()
        
        # Etapa 4: Fazer predições futuras
        logger.info("Etapa 4: Fazendo predições futuras")
        predictions_summary = {}
        
        for city in trained_cities[:3]:  # Fazer predições para as primeiras 3 cidades como exemplo
            try:
                predictions, conf_intervals = predictor.predict_future(city, steps=24)
                predictions_summary[city] = {
                    'next_24h_avg': float(predictions.mean()),
                    'min_predicted': float(predictions.min()),
                    'max_predicted': float(predictions.max()),
                    'predictions_count': len(predictions)
                }
                logger.info(f"Predições para {city}: temp média próximas 24h = {predictions.mean():.2f}°C")
            except Exception as e:
                logger.error(f"Erro ao fazer predições para {city}: {e}")
        
        # # Etapa 5: Salvar modelos
        # logger.info("Etapa 5: Salvando modelos")
        # predictor.save_models()
        # logger.info("Modelos salvos com sucesso")
        
        # Relatório final
        logger.info("="*60)
        logger.info("RELATÓRIO FINAL - PREVISÃO DE TEMPERATURA POR CIDADE")
        logger.info("="*60)
        logger.info(f"Registros processados: {len(df_processed)}")
        logger.info(f"Cidades treinadas: {len(trained_cities)}")
        logger.info(f"Variáveis exógenas utilizadas: {len(exog_columns)}")
        
        # Métricas médias
        if metrics_by_city:
            avg_mae = sum(m['mae'] for m in metrics_by_city.values()) / len(metrics_by_city)
            avg_rmse = sum(m['rmse'] for m in metrics_by_city.values()) / len(metrics_by_city)
            avg_r2 = sum(m['r2'] for m in metrics_by_city.values()) / len(metrics_by_city)
            
            logger.info(f"MAE médio: {avg_mae:.2f}°C")
            logger.info(f"RMSE médio: {avg_rmse:.2f}°C")
            logger.info(f"R² médio: {avg_r2:.4f}")
        
        # Top 3 cidades por performance (R²)
        if metrics_by_city:
            top_cities = sorted(metrics_by_city.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
            logger.info("\nTop 3 cidades por performance (R²):")
            for i, (city, metrics) in enumerate(top_cities, 1):
                logger.info(f"{i}. {city}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.2f}°C")
        
        # Resumo das predições
        if predictions_summary:
            logger.info("\nPredições para próximas 24h (amostra):")
            for city, pred in predictions_summary.items():
                logger.info(f"{city}: {pred['min_predicted']:.1f}°C - {pred['max_predicted']:.1f}°C (média: {pred['next_24h_avg']:.1f}°C)")
        
        logger.info("="*60)
        logger.info("Pipeline de temperatura por cidade executado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na execução do pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
