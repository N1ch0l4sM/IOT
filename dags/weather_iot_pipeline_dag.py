"""
DAG do Airflow para pipeline IoT de dados meteorológicos
"""
import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.weather_fetch import WeatherFetcher
from src.weather_hour import WeatherHourlyAggregator
from src.data_processing.weather_processor import WeatherDataProcessor
from src.ml.city_predictor import CityTemperaturePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configurações padrão do DAG
default_args = {
    'owner': 'iot-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Definição do DAG
dag = DAG(
    'weather_iot_pipeline',
    default_args=default_args,
    description='Pipeline IoT para coleta, processamento e previsão de dados meteorológicos',
    schedule_interval='@hourly',  # Executa a cada hora
    max_active_runs=1,
    tags=['iot', 'weather', 'ml', 'pipeline']
)


def extract_weather_data(**context):
    """
    Task para extrair dados meteorológicos da API
    """
    try:
        logger.info("Iniciando extração de dados meteorológicos")
        
        fetcher = WeatherFetcher()
        stats = fetcher.fetch_all_cities()
        
        # Registrar estatísticas no XCom
        context['task_instance'].xcom_push(key='fetch_stats', value=stats)
        
        if stats['successful_saves'] == 0:
            raise Exception("Nenhum dado foi salvo com sucesso na extração")
        
        logger.info(f"Extração concluída: {stats['successful_saves']} registros salvos")
        return stats
        
    except Exception as e:
        logger.error(f"Erro na extração de dados: {e}")
        raise


def aggregate_hourly_data(**context):
    """
    Task para agregar dados meteorológicos por hora
    """
    try:
        logger.info("Iniciando agregação horária de dados")
        
        aggregator = WeatherHourlyAggregator()
        stats = aggregator.process_hourly_aggregation(hours_back=1)
        
        # Registrar estatísticas no XCom
        context['task_instance'].xcom_push(key='aggregation_stats', value=stats)
        
        if not stats['success']:
            raise Exception(f"Falha na agregação: {stats.get('error_message', 'Erro desconhecido')}")
        
        logger.info(f"Agregação concluída: {stats['records_saved']} registros processados")
        aggregator.close()
        
        return stats
        
    except Exception as e:
        logger.error(f"Erro na agregação de dados: {e}")
        raise


def process_weather_data(**context):
    """
    Task para processar e preparar dados para ML
    """
    try:
        logger.info("Iniciando processamento de dados para ML")
        
        processor = WeatherDataProcessor()
        
        # Carregar dados processados
        df_raw = processor.load_processed_data(source='iot_weather_db')
        logger.info(f"Dados carregados: {len(df_raw)} registros")
        
        # Limpar dados
        df_clean = processor.clean_data(df_raw)
        logger.info(f"Dados limpos: {len(df_clean)} registros")
        
        # Feature engineering
        df_processed = processor.feature_engineering(df_clean)
        logger.info(f"Features criadas: {len(df_processed)} registros")
        
        # Registrar estatísticas no XCom
        stats = {
            'raw_records': len(df_raw),
            'clean_records': len(df_clean),
            'processed_records': len(df_processed)
        }
        context['task_instance'].xcom_push(key='processing_stats', value=stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"Erro no processamento de dados: {e}")
        raise


def train_ml_models(**context):
    """
    Task para treinar modelos de machine learning
    """
    try:
        logger.info("Iniciando treinamento de modelos ML")
        
        # Carregar dados processados
        processor = WeatherDataProcessor()
        df_raw = processor.load_processed_data(source='iot_weather_db')
        df_clean = processor.clean_data(df_raw)
        df_processed = processor.feature_engineering(df_clean)
        
        # Filtrar dados recentes (últimos 30 dias)
        import pandas as pd
        last_days = 30
        df_recent = df_processed[
            df_processed['date'] >= df_processed['date'].max() - pd.Timedelta(days=last_days)
        ].copy()
        
        # Filtrar por algumas cidades específicas para exemplo
        df_filtered = df_recent[
            (df_recent['idcity'] == 22) | 
            (df_recent['idcity'] == 29) | 
            (df_recent['idcity'] == 47)
        ]
        
        if len(df_filtered) < 100:
            logger.warning("Poucos dados disponíveis para treinamento, pulando esta execução")
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
        
        # Avaliar modelos
        metrics_by_city = predictor.evaluate_models()
        
        # Fazer algumas predições de exemplo
        predictions_summary = {}
        for city in trained_cities[:2]:  # Primeiras 2 cidades
            try:
                predictions, conf_intervals = predictor.predict_future(city, steps=24)
                predictions_summary[city] = {
                    'next_24h_avg': float(predictions.mean()),
                    'min_predicted': float(predictions.min()),
                    'max_predicted': float(predictions.max())
                }
            except Exception as e:
                logger.warning(f"Erro ao fazer predições para cidade {city}: {e}")
        
        # Registrar estatísticas no XCom
        stats = {
            'trained_cities': len(trained_cities),
            'records_used': len(df_filtered),
            'metrics': metrics_by_city,
            'predictions': predictions_summary
        }
        context['task_instance'].xcom_push(key='ml_stats', value=stats)
        
        logger.info(f"Treinamento concluído: {len(trained_cities)} cidades treinadas")
        return stats
        
    except Exception as e:
        logger.error(f"Erro no treinamento de modelos: {e}")
        raise


def generate_pipeline_report(**context):
    """
    Task para gerar relatório final do pipeline
    """
    try:
        logger.info("Gerando relatório final do pipeline")
        
        # Recuperar estatísticas das tasks anteriores
        fetch_stats = context['task_instance'].xcom_pull(
            task_ids='extract_weather_data', key='fetch_stats'
        )
        aggregation_stats = context['task_instance'].xcom_pull(
            task_ids='aggregate_hourly_data', key='aggregation_stats'
        )
        processing_stats = context['task_instance'].xcom_pull(
            task_ids='process_weather_data', key='processing_stats'
        )
        ml_stats = context['task_instance'].xcom_pull(
            task_ids='train_ml_models', key='ml_stats'
        )
        
        # Gerar relatório
        report = {
            'execution_date': context['execution_date'].isoformat(),
            'dag_run_id': context['dag_run'].run_id,
            'extraction': fetch_stats,
            'aggregation': aggregation_stats,
            'processing': processing_stats,
            'machine_learning': ml_stats,
            'pipeline_status': 'SUCCESS'
        }
        
        logger.info("="*60)
        logger.info("RELATÓRIO FINAL DO PIPELINE IoT")
        logger.info("="*60)
        logger.info(f"Data de execução: {report['execution_date']}")
        logger.info(f"Run ID: {report['dag_run_id']}")
        
        if fetch_stats:
            logger.info(f"Extração: {fetch_stats['successful_saves']} registros salvos")
        
        if aggregation_stats:
            logger.info(f"Agregação: {aggregation_stats['records_saved']} registros processados")
        
        if processing_stats:
            logger.info(f"Processamento: {processing_stats['processed_records']} registros")
        
        if ml_stats:
            logger.info(f"ML: {ml_stats['trained_cities']} cidades treinadas")
        
        logger.info("Pipeline executado com sucesso!")
        logger.info("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Erro na geração do relatório: {e}")
        raise


# Definição das tasks
extract_task = PythonOperator(
    task_id='extract_weather_data',
    python_callable=extract_weather_data,
    dag=dag,
    doc_md="""
    ### Extração de Dados Meteorológicos
    
    Esta task coleta dados meteorológicos da API OpenWeatherMap para todas as cidades
    configuradas e salva os dados brutos no MongoDB.
    
    **Responsabilidades:**
    - Buscar dados da API OpenWeatherMap
    - Validar resposta da API
    - Salvar dados no MongoDB
    - Registrar estatísticas de coleta
    """
)

aggregate_task = PythonOperator(
    task_id='aggregate_hourly_data',
    python_callable=aggregate_hourly_data,
    dag=dag,
    doc_md="""
    ### Agregação Horária de Dados
    
    Esta task processa os dados coletados no MongoDB, agrega por hora e localização,
    e salva os resultados no PostgreSQL.
    
    **Responsabilidades:**
    - Carregar dados do MongoDB
    - Agregar dados por coordenadas
    - Fazer join com dados das cidades
    - Salvar no PostgreSQL
    """
)

process_task = PythonOperator(
    task_id='process_weather_data',
    python_callable=process_weather_data,
    dag=dag,
    doc_md="""
    ### Processamento de Dados para ML
    
    Esta task prepara os dados para machine learning, incluindo limpeza,
    validação e feature engineering.
    
    **Responsabilidades:**
    - Carregar dados do PostgreSQL
    - Limpar e validar dados
    - Criar features para ML
    - Validar qualidade dos dados
    """
)

ml_task = PythonOperator(
    task_id='train_ml_models',
    python_callable=train_ml_models,
    dag=dag,
    doc_md="""
    ### Treinamento de Modelos ML
    
    Esta task treina modelos de machine learning para previsão de temperatura
    por cidade usando dados históricos.
    
    **Responsabilidades:**
    - Preparar dados para treinamento
    - Treinar modelos SARIMAX por cidade
    - Avaliar performance dos modelos
    - Fazer predições de exemplo
    """
)

report_task = PythonOperator(
    task_id='generate_pipeline_report',
    python_callable=generate_pipeline_report,
    dag=dag,
    doc_md="""
    ### Geração de Relatório
    
    Esta task gera um relatório final consolidando todas as estatísticas
    e resultados do pipeline.
    
    **Responsabilidades:**
    - Coletar estatísticas de todas as tasks
    - Gerar relatório consolidado
    - Registrar logs estruturados
    - Validar sucesso geral do pipeline
    """
)

# Dependências das tasks
extract_task >> aggregate_task >> process_task >> ml_task >> report_task
