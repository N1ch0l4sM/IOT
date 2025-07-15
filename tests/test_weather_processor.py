"""
Testes unitários para o processamento de dados
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_processing.weather_processor import WeatherDataProcessor

class TestWeatherDataProcessor(unittest.TestCase):
    """Testes para WeatherDataProcessor"""
    
    def setUp(self):
        """Configurar teste"""
        self.processor = WeatherDataProcessor()
    
    def test_generate_sample_data(self):
        """Testar geração de dados de exemplo"""
        df = self.processor._generate_sample_data(n_samples=100)
        
        # Verificar se DataFrame foi criado
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        
        # Verificar colunas necessárias
        required_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_clean_data(self):
        """Testar limpeza de dados"""
        # Criar DataFrame com dados problemáticos
        df = pd.DataFrame({
            'temperature': [25.0, np.nan, 30.0, 1000.0, 20.0],
            'humidity': [60.0, 70.0, np.nan, 50.0, 80.0],
            'pressure': [1013.0, 1015.0, 1010.0, 1012.0, np.nan],
            'wind_speed': [10.0, 15.0, 5.0, 8.0, 12.0],
            'precipitation': [0.0, 2.0, 0.0, 1.0, 0.0]
        })
        
        # Limpar dados
        df_clean = self.processor.clean_data(df)
        
        # Verificar se dados foram limpos
        self.assertFalse(df_clean.isnull().any().any())
        self.assertTrue(len(df_clean) > 0)
    
    def test_feature_engineering(self):
        """Testar criação de features"""
        # Criar DataFrame base
        df = pd.DataFrame({
            'temperature': [25.0, 30.0, 20.0],
            'humidity': [60.0, 70.0, 50.0],
            'pressure': [1013.0, 1015.0, 1010.0],
            'wind_speed': [10.0, 15.0, 5.0],
            'precipitation': [0.0, 2.0, 0.0],
            'recorded_at': pd.date_range('2023-01-01', periods=3, freq='H')
        })
        
        # Criar features
        df_features = self.processor.feature_engineering(df)
        
        # Verificar se features foram criadas
        expected_features = ['feels_like', 'dew_point', 'will_rain', 'hour', 'month']
        for feature in expected_features:
            self.assertIn(feature, df_features.columns)
    
    def test_handle_missing_values(self):
        """Testar tratamento de valores ausentes"""
        # Criar DataFrame com valores ausentes
        df = pd.DataFrame({
            'temperature': [25.0, np.nan, 30.0],
            'humidity': [60.0, 70.0, np.nan],
            'location': ['SP', None, 'RJ']
        })
        
        # Tratar valores ausentes
        df_handled = self.processor._handle_missing_values(df)
        
        # Verificar se valores foram preenchidos
        self.assertFalse(df_handled.isnull().any().any())
    
    def test_remove_outliers(self):
        """Testar remoção de outliers"""
        # Criar DataFrame com outliers
        df = pd.DataFrame({
            'temperature': [25.0, 30.0, 20.0, 1000.0, 22.0],  # 1000.0 é outlier
            'humidity': [60.0, 70.0, 50.0, 65.0, 55.0],
            'pressure': [1013.0, 1015.0, 1010.0, 1012.0, 1014.0]
        })
        
        original_len = len(df)
        df_no_outliers = self.processor._remove_outliers(df)
        
        # Verificar se outliers foram removidos
        self.assertLess(len(df_no_outliers), original_len)
        self.assertNotIn(1000.0, df_no_outliers['temperature'].values)

if __name__ == '__main__':
    unittest.main()
