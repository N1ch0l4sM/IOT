"""
DAG principal do pipeline de previsão de chuva
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Adicionar src ao path
sys.path.append('/opt/airflow/src')

from src.data_processing.weather_processor import WeatherDataProcessor
from src.ml.rain_predictor import RainPredictor, retrain_model
from src.utils.logger import setup_logging

# Configurar logging
logger = setup_logging()

# Argumentos padrão do DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definir DAG
dag = DAG(
    'weather_prediction_pipeline',
    default_args=default_args,
    description='Pipeline completo para previsão de chuva',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['weather', 'ml', 'iot']
)

def extract_weather_data(**context):
    """Extrai dados climáticos"""
    try:
        processor = WeatherDataProcessor()
        
        # Carregar dados brutos
        df = processor.load_raw_data(source='kaggle')
        
        # Salvar dados brutos no MinIO
        processor.save_processed_data(df, destination='minio')
        
        logger.info(f"Dados extraídos: {len(df)} registros")
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Erro na extração: {e}")
        raise

def transform_weather_data(**context):
    """Transforma e limpa dados climáticos"""
    try:
        processor = WeatherDataProcessor()
        
        # Carregar dados brutos
        df = processor.load_raw_data(source='minio')
        
        # Limpar dados
        df_clean = processor.clean_data(df)
        
        # Feature engineering
        df_processed = processor.feature_engineering(df_clean)
        
        # Salvar dados processados
        processor.save_processed_data(df_processed, destination='both')
        
        logger.info(f"Dados transformados: {len(df_processed)} registros")
        
        return len(df_processed)
        
    except Exception as e:
        logger.error(f"Erro na transformação: {e}")
        raise

def train_ml_model(**context):
    """Treina modelo de machine learning"""
    try:
        # Verificar se é hora de retreinar (a cada 24 horas)
        execution_date = context['execution_date']
        if execution_date.hour != 0:
            logger.info("Pulando treinamento - não é hora de retreinar")
            return "skipped"
        
        # Retreinar modelo
        retrain_model()
        
        logger.info("Modelo treinado com sucesso")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise

def make_predictions(**context):
    """Faz predições com modelo treinado"""
    try:
        from src.utils.database import db_connection
        
        # Carregar dados recentes
        query = """
        SELECT * FROM weather_data 
        WHERE recorded_at >= NOW() - INTERVAL '1 hour'
        AND id NOT IN (SELECT weather_data_id FROM predictions WHERE predicted_at >= NOW() - INTERVAL '1 hour')
        """
        df = db_connection.execute_query(query)
        
        if len(df) == 0:
            logger.info("Nenhum dado novo para predição")
            return "no_data"
        
        # Carregar modelo
        predictor = RainPredictor()
        predictor.load_model()
        
        # Fazer predições
        predictions, probabilities = predictor.predict(df)
        
        # Salvar predições no banco
        predictions_df = df[['id']].copy()
        predictions_df['weather_data_id'] = df['id']
        predictions_df['prediction'] = predictions
        predictions_df['confidence'] = probabilities
        predictions_df['model_version'] = '1.0.0'
        predictions_df['predicted_at'] = datetime.now()
        
        db_connection.insert_dataframe(
            predictions_df[['weather_data_id', 'prediction', 'confidence', 'model_version', 'predicted_at']], 
            'predictions'
        )
        
        logger.info(f"Predições feitas para {len(df)} registros")
        
        return len(df)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise

def data_quality_check(**context):
    """Verifica qualidade dos dados"""
    try:
        from src.utils.database import db_connection
        
        # Verificar dados recentes
        query = """
        SELECT COUNT(*) as total_records,
               COUNT(CASE WHEN temperature IS NULL THEN 1 END) as null_temperature,
               COUNT(CASE WHEN humidity IS NULL THEN 1 END) as null_humidity,
               COUNT(CASE WHEN pressure IS NULL THEN 1 END) as null_pressure
        FROM weather_data 
        WHERE recorded_at >= NOW() - INTERVAL '1 hour'
        """
        
        result = db_connection.execute_query(query)
        
        total_records = result.iloc[0]['total_records']
        null_temperature = result.iloc[0]['null_temperature']
        null_humidity = result.iloc[0]['null_humidity']
        null_pressure = result.iloc[0]['null_pressure']
        
        # Verificar se qualidade está adequada
        if total_records == 0:
            logger.warning("Nenhum dado recente encontrado")
            return "no_data"
        
        null_percentage = (null_temperature + null_humidity + null_pressure) / (total_records * 3) * 100
        
        if null_percentage > 10:
            logger.warning(f"Qualidade dos dados baixa: {null_percentage:.2f}% valores nulos")
            return "low_quality"
        
        logger.info(f"Qualidade dos dados OK: {null_percentage:.2f}% valores nulos")
        
        return "quality_ok"
        
    except Exception as e:
        logger.error(f"Erro na verificação de qualidade: {e}")
        raise

# Definir tasks
extract_task = PythonOperator(
    task_id='extract_weather_data',
    python_callable=extract_weather_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_weather_data',
    python_callable=transform_weather_data,
    dag=dag,
)

quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_ml_model',
    python_callable=train_ml_model,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag,
)

# Definir dependências
extract_task >> transform_task >> quality_check_task >> [train_model_task, predict_task]
