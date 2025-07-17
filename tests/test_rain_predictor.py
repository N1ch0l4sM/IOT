"""
Testes unitários para o modelo de machine learning
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml.rain_predictor import RainPredictor
class TestRainPredictor(unittest.TestCase):
    """Testes para RainPredictor"""
    
    def setUp(self):
        """Configurar teste"""
        self.predictor = RainPredictor()
    
    def test_prepare_features(self):
        """Testar preparação de features"""
        # Criar DataFrame com features
        df = pd.DataFrame({
            'temperature': [25.0, 30.0, 20.0],
            'humidity': [60.0, 70.0, 50.0],
            'pressure': [1013.0, 1015.0, 1010.0],
            'wind_speed': [10.0, 15.0, 5.0],
            'feels_like': [27.0, 32.0, 22.0],
            'dew_point': [15.0, 20.0, 10.0],
            'pressure_tendency': [0.0, 2.0, -3.0],
            'temp_humidity_interaction': [1500.0, 2100.0, 1000.0],
            'wind_pressure_interaction': [10130.0, 15225.0, 5050.0],
            'hour': [12, 15, 8],
            'day_of_week': [1, 2, 3],
            'month': [6, 7, 8],
            'wind_direction': ['N', 'E', 'S']
        })
        
        # Preparar features
        X = self.predictor.prepare_features(df)
        
        # Verificar se features foram preparadas
        self.assertIsInstance(X, pd.DataFrame)
        self.assertGreater(len(X.columns), 0)
        self.assertEqual(len(X), len(df))
    
    def test_train_model(self):
        """Testar treinamento do modelo"""
        # Criar dados de treino
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'temperature': np.random.normal(25, 5, n_samples),
            'humidity': np.random.normal(65, 15, n_samples),
            'pressure': np.random.normal(1013, 10, n_samples),
            'wind_speed': np.random.exponential(8, n_samples),
            'feels_like': np.random.normal(27, 5, n_samples),
            'dew_point': np.random.normal(15, 5, n_samples),
            'pressure_tendency': np.random.normal(0, 2, n_samples),
            'temp_humidity_interaction': np.random.normal(1500, 300, n_samples),
            'wind_pressure_interaction': np.random.normal(8000, 1000, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'rain_probability': np.random.binomial(1, 0.3, n_samples)
        })
        
        # Treinar modelo
        metrics = self.predictor.train_model(df)
        
        # Verificar se modelo foi treinado
        self.assertIsNotNone(self.predictor.model)
        self.assertIsInstance(metrics, dict)
        self.assertIn('auc', metrics)
        self.assertGreater(metrics['auc'], 0)
        self.assertLess(metrics['auc'], 1)
    
    def test_predict(self):
        """Testar predições"""
        # Primeiro treinar um modelo simples
        np.random.seed(42)
        n_samples = 100
        
        df_train = pd.DataFrame({
            'temperature': np.random.normal(25, 5, n_samples),
            'humidity': np.random.normal(65, 15, n_samples),
            'pressure': np.random.normal(1013, 10, n_samples),
            'wind_speed': np.random.exponential(8, n_samples),
            'feels_like': np.random.normal(27, 5, n_samples),
            'dew_point': np.random.normal(15, 5, n_samples),
            'pressure_tendency': np.random.normal(0, 2, n_samples),
            'temp_humidity_interaction': np.random.normal(1500, 300, n_samples),
            'wind_pressure_interaction': np.random.normal(8000, 1000, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'rain_probability': np.random.binomial(1, 0.3, n_samples)
        })
        
        self.predictor.train_model(df_train)
        
        # Criar dados para predição
        df_predict = df_train.head(10).drop('rain_probability', axis=1)
        
        # Fazer predições
        predictions, probabilities = self.predictor.predict(df_predict)
        
        # Verificar predições
        self.assertEqual(len(predictions), len(df_predict))
        self.assertEqual(len(probabilities), len(df_predict))
        self.assertTrue(all(p in [0, 1] for p in predictions))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
    
    def test_get_feature_importance(self):
        """Testar importância das features"""
        # Treinar modelo simples
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'temperature': np.random.normal(25, 5, n_samples),
            'humidity': np.random.normal(65, 15, n_samples),
            'pressure': np.random.normal(1013, 10, n_samples),
            'wind_speed': np.random.exponential(8, n_samples),
            'feels_like': np.random.normal(27, 5, n_samples),
            'dew_point': np.random.normal(15, 5, n_samples),
            'pressure_tendency': np.random.normal(0, 2, n_samples),
            'temp_humidity_interaction': np.random.normal(1500, 300, n_samples),
            'wind_pressure_interaction': np.random.normal(8000, 1000, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'rain_probability': np.random.binomial(1, 0.3, n_samples)
        })
        
        self.predictor.train_model(df)
        
        # Obter importância
        importance_df = self.predictor.get_feature_importance()
        
        # Verificar importância
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        self.assertGreater(len(importance_df), 0)

if __name__ == '__main__':
    unittest.main()
