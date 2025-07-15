"""
Script para gerar dados de exemplo do Kaggle para demonstração
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_weather_dataset(n_samples=10000, output_path='data/raw/weather_data.csv'):
    """
    Gera dataset de exemplo simulando dados do Kaggle
    """
    np.random.seed(42)
    
    # Gerar datas
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    # Localizações brasileiras
    locations = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Fortaleza']
    
    # Gerar dados base
    data = []
    
    for i, date in enumerate(dates):
        location = np.random.choice(locations)
        
        # Padrões sazonais
        day_of_year = date.timetuple().tm_yday
        seasonal_temp = 25 + 5 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Padrões diários
        hour_factor = np.sin(2 * np.pi * date.hour / 24)
        
        # Temperatura com padrão sazonal e diário
        temperature = seasonal_temp + 3 * hour_factor + np.random.normal(0, 2)
        
        # Umidade (inversamente relacionada à temperatura)
        humidity = 80 - 0.5 * temperature + np.random.normal(0, 10)
        humidity = np.clip(humidity, 20, 100)
        
        # Pressão atmosférica
        pressure = 1013 + np.random.normal(0, 15)
        
        # Velocidade do vento
        wind_speed = np.random.exponential(8) + np.random.normal(0, 2)
        wind_speed = np.clip(wind_speed, 0, 50)
        
        # Direção do vento
        wind_direction = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        # Precipitação (correlacionada com umidade e pressão baixa)
        rain_probability = (humidity - 50) / 50 + (1013 - pressure) / 30
        rain_probability = np.clip(rain_probability, 0, 1)
        
        will_rain = np.random.random() < rain_probability
        precipitation = np.random.exponential(2) if will_rain else 0
        
        # Qualidade do ar (exemplo)
        air_quality = np.random.choice(['Good', 'Moderate', 'Poor'], p=[0.6, 0.3, 0.1])
        
        # Visibilidade
        visibility = 10 - (humidity / 20) + np.random.normal(0, 1)
        visibility = np.clip(visibility, 1, 15)
        
        data.append({
            'datetime': date,
            'location': location,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': wind_direction,
            'precipitation': round(precipitation, 2),
            'visibility': round(visibility, 1),
            'air_quality': air_quality,
            'season': get_season(date.month),
            'is_weekend': date.weekday() >= 5,
            'hour': date.hour,
            'month': date.month,
            'day_of_week': date.weekday()
        })
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Adicionar algumas features derivadas
    df['feels_like'] = df['temperature'] + 0.5 * (df['humidity'] / 100) * (df['temperature'] - 14.5)
    df['dew_point'] = df['temperature'] - ((100 - df['humidity']) / 5)
    df['heat_index'] = calculate_heat_index(df['temperature'], df['humidity'])
    df['wind_chill'] = calculate_wind_chill(df['temperature'], df['wind_speed'])
    
    # Criar target
    df['will_rain'] = (df['precipitation'] > 0).astype(int)
    
    # Salvar arquivo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset gerado com {len(df)} registros")
    print(f"Arquivo salvo em: {output_path}")
    print(f"Período: {df['datetime'].min()} até {df['datetime'].max()}")
    print(f"Porcentagem de chuva: {df['will_rain'].mean()*100:.1f}%")
    
    return df

def get_season(month):
    """Retorna estação do ano (Hemisfério Sul)"""
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

def calculate_heat_index(temp, humidity):
    """Calcula índice de calor"""
    # Fórmula simplificada
    hi = temp + 0.5 * (humidity / 100) * (temp - 14.5)
    return np.round(hi, 1)

def calculate_wind_chill(temp, wind_speed):
    """Calcula sensação térmica com vento"""
    # Fórmula aplicável para temperaturas baixas
    wc = np.where(
        temp <= 10,
        13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16),
        temp
    )
    return np.round(wc, 1)

if __name__ == "__main__":
    # Gerar dataset
    df = generate_weather_dataset()
    
    # Mostrar informações básicas
    print("\n" + "="*50)
    print("INFORMAÇÕES DO DATASET")
    print("="*50)
    print(df.info())
    print("\n" + "="*50)
    print("ESTATÍSTICAS DESCRITIVAS")
    print("="*50)
    print(df.describe())
    print("\n" + "="*50)
    print("DISTRIBUIÇÃO POR LOCALIZAÇÃO")
    print("="*50)
    print(df['location'].value_counts())
    print("\n" + "="*50)
    print("DISTRIBUIÇÃO POR ESTAÇÃO")
    print("="*50)
    print(df['season'].value_counts())
