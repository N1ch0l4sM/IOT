"""
Dashboard principal do Streamlit para visualização de dados e predições
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append('/app/src')

from src.config import STREAMLIT_CONFIG
from src.utils.database import db_connection
from src.ml.rain_predictor import RainPredictor
from src.utils.logger import setup_logging

# Configurar logging
logger = setup_logging()

# Configuração da página
st.set_page_config(
    page_title=STREAMLIT_CONFIG['title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title(f"{STREAMLIT_CONFIG['page_icon']} {STREAMLIT_CONFIG['title']}")

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Filtros de data
date_range = st.sidebar.date_input(
    "Período de análise",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
    max_value=datetime.now()
)

# Filtro de localização
locations = ['Todos', 'São Paulo', 'Rio de Janeiro', 'Belo Horizonte']
selected_location = st.sidebar.selectbox("Localização", locations)

# Atualização automática
auto_refresh = st.sidebar.checkbox("Atualização automática", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Intervalo (segundos)", 30, 300, 60)

st.sidebar.markdown("---")

# Funções para carregar dados
@st.cache_data(ttl=60)
def load_weather_data(start_date, end_date, location=None):
    """Carrega dados climáticos"""
    query = """
    SELECT * FROM weather_data 
    WHERE recorded_at BETWEEN %s AND %s
    """
    params = [start_date, end_date]
    
    if location and location != 'Todos':
        query += " AND location = %s"
        params.append(location)
    
    query += " ORDER BY recorded_at DESC"
    
    return db_connection.execute_query(query, params)

@st.cache_data(ttl=60)
def load_predictions_data(start_date, end_date):
    """Carrega dados de predições"""
    query = """
    SELECT p.*, w.location, w.temperature, w.humidity, w.recorded_at
    FROM predictions p
    JOIN weather_data w ON p.weather_data_id = w.id
    WHERE p.predicted_at BETWEEN %s AND %s
    ORDER BY p.predicted_at DESC
    """
    
    return db_connection.execute_query(query, [start_date, end_date])

@st.cache_data(ttl=300)
def load_model_metrics():
    """Carrega métricas do modelo"""
    query = """
    SELECT * FROM model_metrics 
    ORDER BY trained_at DESC 
    LIMIT 1
    """
    
    try:
        return db_connection.execute_query(query)
    except:
        return pd.DataFrame()

# Carregar dados
try:
    start_date, end_date = date_range
    
    # Dados climáticos
    weather_df = load_weather_data(start_date, end_date, selected_location)
    
    # Dados de predições
    predictions_df = load_predictions_data(start_date, end_date)
    
    # Métricas do modelo
    model_metrics = load_model_metrics()
    
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_records = len(weather_df)
    st.metric("Total de Registros", total_records)

with col2:
    if not predictions_df.empty:
        rain_predictions = predictions_df['prediction'].sum()
        rain_percentage = (rain_predictions / len(predictions_df)) * 100
        st.metric("Previsões de Chuva", f"{rain_percentage:.1f}%")
    else:
        st.metric("Previsões de Chuva", "0%")

with col3:
    if not weather_df.empty:
        avg_temp = weather_df['temperature'].mean()
        st.metric("Temperatura Média", f"{avg_temp:.1f}°C")
    else:
        st.metric("Temperatura Média", "N/A")

with col4:
    if not model_metrics.empty:
        model_auc = model_metrics.iloc[0]['auc']
        st.metric("AUC do Modelo", f"{model_auc:.3f}")
    else:
        st.metric("AUC do Modelo", "N/A")

st.markdown("---")

# Abas principais
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise Climática", "🔮 Predições", "🤖 Modelo ML", "📈 Monitoramento"])

with tab1:
    st.header("Análise de Dados Climáticos")
    
    if not weather_df.empty:
        # Gráfico de temperatura ao longo do tempo
        fig_temp = px.line(
            weather_df, 
            x='recorded_at', 
            y='temperature',
            color='location' if selected_location == 'Todos' else None,
            title='Temperatura ao Longo do Tempo',
            labels={'recorded_at': 'Data/Hora', 'temperature': 'Temperatura (°C)'}
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Gráficos de umidade e pressão
        col1, col2 = st.columns(2)
        
        with col1:
            fig_humidity = px.line(
                weather_df,
                x='recorded_at',
                y='humidity',
                title='Umidade Relativa',
                labels={'recorded_at': 'Data/Hora', 'humidity': 'Umidade (%)'}
            )
            st.plotly_chart(fig_humidity, use_container_width=True)
        
        with col2:
            fig_pressure = px.line(
                weather_df,
                x='recorded_at',
                y='pressure',
                title='Pressão Atmosférica',
                labels={'recorded_at': 'Data/Hora', 'pressure': 'Pressão (hPa)'}
            )
            st.plotly_chart(fig_pressure, use_container_width=True)
        
        # Distribuição das variáveis
        st.subheader("Distribuição das Variáveis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist_temp = px.histogram(
                weather_df,
                x='temperature',
                nbins=30,
                title='Distribuição da Temperatura',
                labels={'temperature': 'Temperatura (°C)', 'count': 'Frequência'}
            )
            st.plotly_chart(fig_dist_temp, use_container_width=True)
        
        with col2:
            fig_dist_humidity = px.histogram(
                weather_df,
                x='humidity',
                nbins=30,
                title='Distribuição da Umidade',
                labels={'humidity': 'Umidade (%)', 'count': 'Frequência'}
            )
            st.plotly_chart(fig_dist_humidity, use_container_width=True)
        
        # Correlação entre variáveis
        st.subheader("Correlação entre Variáveis")
        
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        available_columns = [col for col in numeric_columns if col in weather_df.columns]
        
        if len(available_columns) > 1:
            corr_matrix = weather_df[available_columns].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matriz de Correlação",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    else:
        st.info("Nenhum dado climático disponível para o período selecionado.")

with tab2:
    st.header("Predições de Chuva")
    
    if not predictions_df.empty:
        # Predições ao longo do tempo
        fig_predictions = px.scatter(
            predictions_df,
            x='predicted_at',
            y='confidence',
            color='prediction',
            title='Predições de Chuva ao Longo do Tempo',
            labels={
                'predicted_at': 'Data/Hora da Predição',
                'confidence': 'Confiança',
                'prediction': 'Predição'
            }
        )
        st.plotly_chart(fig_predictions, use_container_width=True)
        
        # Estatísticas das predições
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Estatísticas das Predições")
            rain_count = predictions_df['prediction'].sum()
            total_predictions = len(predictions_df)
            
            stats_df = pd.DataFrame({
                'Métrica': ['Total de Predições', 'Predições de Chuva', 'Predições Sem Chuva', 'Taxa de Chuva'],
                'Valor': [
                    total_predictions,
                    rain_count,
                    total_predictions - rain_count,
                    f"{(rain_count / total_predictions) * 100:.1f}%"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Distribuição de Confiança")
            fig_confidence = px.histogram(
                predictions_df,
                x='confidence',
                nbins=20,
                title='Distribuição da Confiança nas Predições',
                labels={'confidence': 'Confiança', 'count': 'Frequência'}
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Tabela de predições recentes
        st.subheader("Predições Recentes")
        recent_predictions = predictions_df.head(10)[
            ['predicted_at', 'location', 'prediction', 'confidence', 'temperature', 'humidity']
        ]
        recent_predictions['prediction'] = recent_predictions['prediction'].map({True: '🌧️ Chuva', False: '☀️ Sem Chuva'})
        recent_predictions['confidence'] = recent_predictions['confidence'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(recent_predictions, use_container_width=True)
    
    else:
        st.info("Nenhuma predição disponível para o período selecionado.")

with tab3:
    st.header("Modelo de Machine Learning")
    
    # Informações do modelo
    if not model_metrics.empty:
        st.subheader("Métricas do Modelo")
        
        metrics = model_metrics.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AUC", f"{metrics['auc']:.3f}")
            st.metric("Precisão", f"{metrics['precision']:.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        with col3:
            st.metric("Acurácia", f"{metrics['accuracy']:.3f}")
            st.metric("Última Atualização", metrics['trained_at'].strftime("%d/%m/%Y %H:%M"))
    
    else:
        st.info("Nenhuma métrica de modelo disponível.")
    
    # Importância das features
    st.subheader("Importância das Features")
    
    try:
        predictor = RainPredictor()
        predictor.load_model()
        
        importance_df = predictor.get_feature_importance()
        
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Importância das Features',
            labels={'importance': 'Importância', 'feature': 'Feature'}
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao carregar importância das features: {e}")
    
    # Simulador de predição
    st.subheader("Simulador de Predição")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            temp = st.slider("Temperatura (°C)", -10, 50, 25)
            humidity = st.slider("Umidade (%)", 0, 100, 65)
            pressure = st.slider("Pressão (hPa)", 950, 1050, 1013)
        
        with col2:
            wind_speed = st.slider("Velocidade do Vento (km/h)", 0, 100, 10)
            hour = st.slider("Hora do Dia", 0, 23, 12)
            month = st.slider("Mês", 1, 12, datetime.now().month)
        
        submitted = st.form_submit_button("Fazer Predição")
        
        if submitted:
            try:
                # Criar DataFrame com dados de entrada
                input_data = pd.DataFrame({
                    'temperature': [temp],
                    'humidity': [humidity],
                    'pressure': [pressure],
                    'wind_speed': [wind_speed],
                    'hour': [hour],
                    'month': [month],
                    'feels_like': [temp + 0.5 * (humidity / 100) * (temp - 14.5)],
                    'dew_point': [temp - ((100 - humidity) / 5)],
                    'pressure_tendency': [0],
                    'temp_humidity_interaction': [temp * humidity],
                    'wind_pressure_interaction': [wind_speed * pressure],
                    'day_of_week': [datetime.now().weekday()],
                    'wind_direction_encoded': [0]
                })
                
                predictor = RainPredictor()
                predictor.load_model()
                
                prediction, confidence = predictor.predict(input_data)
                
                if prediction[0]:
                    st.success(f"🌧️ Predição: CHUVA (Confiança: {confidence[0]:.3f})")
                else:
                    st.info(f"☀️ Predição: SEM CHUVA (Confiança: {1-confidence[0]:.3f})")
                
            except Exception as e:
                st.error(f"Erro na predição: {e}")

with tab4:
    st.header("Monitoramento do Sistema")
    
    # Status dos serviços
    st.subheader("Status dos Serviços")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            # Testar conexão com banco
            db_connection.execute_query("SELECT 1")
            st.success("✅ Banco de Dados")
        except:
            st.error("❌ Banco de Dados")
    
    with col2:
        try:
            # Testar conexão com MinIO
            from src.utils.minio_client import minio_connection
            minio_connection.list_objects()
            st.success("✅ MinIO")
        except:
            st.error("❌ MinIO")
    
    with col3:
        try:
            # Testar modelo
            predictor = RainPredictor()
            predictor.load_model()
            st.success("✅ Modelo ML")
        except:
            st.error("❌ Modelo ML")
    
    # Logs recentes
    st.subheader("Logs Recentes")
    
    # Aqui você pode implementar a leitura de logs
    st.info("Funcionalidade de logs será implementada em versão futura.")
    
    # Estatísticas do sistema
    st.subheader("Estatísticas do Sistema")
    
    if not weather_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Dados por hora
            weather_df['hour'] = pd.to_datetime(weather_df['recorded_at']).dt.hour
            hourly_data = weather_df.groupby('hour').size().reset_index(name='count')
            
            fig_hourly = px.bar(
                hourly_data,
                x='hour',
                y='count',
                title='Dados por Hora do Dia',
                labels={'hour': 'Hora', 'count': 'Quantidade'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Dados por localização
            if 'location' in weather_df.columns:
                location_data = weather_df.groupby('location').size().reset_index(name='count')
                
                fig_location = px.pie(
                    location_data,
                    values='count',
                    names='location',
                    title='Distribuição por Localização'
                )
                st.plotly_chart(fig_location, use_container_width=True)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Dashboard desenvolvido com Streamlit para o Pipeline IoT de Previsão de Chuvas*")
