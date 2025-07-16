"""
Arquivo principal para executar o pipeline localmente
"""
import sys
import os
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logging
from src.data_processing.weather_processor import WeatherDataProcessor
from src.ml.rain_predictor import RainPredictor, retrain_model

def main():
    """Função principal para executar o pipeline"""
    
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando pipeline de previsão de chuvas")
    
    try:
        # Etapa 1: Processamento de dados
        logger.info("Etapa 1: Processamento de dados")
        processor = WeatherDataProcessor()
        
        # Carregar dados brutos
        df_raw = processor.load_raw_data(source='kaggle')
        logger.info(f"Dados brutos carregados: {len(df_raw)} registros")
        
        # Limpar dados
        df_clean = processor.clean_data(df_raw)
        logger.info(f"Dados limpos: {len(df_clean)} registros")
        
        # Feature engineering
        df_processed = processor.feature_engineering(df_clean)
        logger.info(f"Features criadas: {len(df_processed)} registros")
        
        # Salvar dados processados
       # processor.save_processed_data(df_processed, destination='both')
        #logger.info("Dados salvos com sucesso")
        
        # Etapa 2: Treinamento do modelo
        logger.info("Etapa 2: Treinamento do modelo")
        predictor = RainPredictor()
        
        # Treinar modelo
        metrics = predictor.train_model(df_processed)
        logger.info(f"Modelo treinado - AUC: {metrics['auc']:.4f}")
        
        # Salvar modelo
        #predictor.save_model()
        #logger.info("Modelo salvo com sucesso")
        
        # Etapa 3: Fazer predições
        logger.info("Etapa 3: Fazendo predições")
        
        # Usar dados recentes para predição
        recent_data = df_processed.tail(100)
        predictions, probabilities = predictor.predict(recent_data)
        
        rain_predictions = sum(predictions)
        logger.info(f"Predições feitas: {rain_predictions}/{len(predictions)} previsões de chuva")
        
        # Relatório final
        logger.info("="*50)
        logger.info("RELATÓRIO FINAL")
        logger.info("="*50)
        logger.info(f"Registros processados: {len(df_processed)}")
        logger.info(f"AUC do modelo: {metrics['auc']:.4f}")
        logger.info(f"Precisão: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Predições de chuva: {rain_predictions}/{len(predictions)}")
        logger.info("="*50)
        
        logger.info("Pipeline executado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na execução do pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
