"""
Modelos de Machine Learning para previsão de temperatura
"""
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from src.utils.database import db_connection
from src.utils.minio_client import minio_connection

logger = logging.getLogger(__name__)

class TemperaturePredictor:
    """Classe para treinamento e predição de temperatura"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest' ou 'sarimax'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.model_metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para o modelo"""
        try:
            logger.info("Preparando features para o modelo de temperatura")
            
            if self.model_type == 'sarimax':
                # Para SARIMAX, usar apenas variáveis exógenas
                feature_columns = ['humidity', 'pressure', 'wind_speed']
            else:
                # Para Random Forest, usar todas as features disponíveis
                feature_columns = [
                    'humidity', 'pressure', 'wind_speed', 'precipitation',
                    'dew_point', 'pressure_tendency',
                    'temp_humidity_interaction', 'wind_pressure_interaction',
                    'hour', 'day_of_week', 'month'
                ]
            
            # Filtrar apenas colunas que existem no DataFrame
            available_columns = [col for col in feature_columns if col in df.columns]
            
            # Adicionar encoding para variáveis categóricas se existirem
            categorical_cols = ['wind_direction', 'season']
            for col in categorical_cols:
                if col in df.columns:
                    df[f'{col}_encoded'] = self.label_encoder.fit_transform(df[col])
                    available_columns.append(f'{col}_encoded')
            
            self.feature_columns = available_columns
            
            # Retornar apenas as features selecionadas
            return df[available_columns]
            
        except Exception as e:
            logger.error(f"Erro ao preparar features: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'temperature') -> Dict:
        """Treina o modelo de previsão de temperatura"""
        try:
            logger.info(f"Iniciando treinamento do modelo {self.model_type}")
            
            if self.model_type == 'sarimax':
                return self._train_sarimax(df, target_column)
            else:
                return self._train_random_forest(df, target_column)
                
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            raise
    
    def _train_random_forest(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Treina modelo Random Forest"""
        # Preparar features
        X = self.prepare_features(df)
        y = df[target_column]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Criar pipeline com normalização e modelo
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Treinar modelo
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        metrics = self._evaluate_random_forest(X_test, y_test)
        self.model_metrics = metrics
        
        logger.info(f"Modelo Random Forest treinado com R²: {metrics['r2']:.4f}")
        return metrics
    
    def _train_sarimax(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Treina modelo SARIMAX"""
        # Preparar dados
        df_sorted = df.sort_values('date').copy()
        y = df_sorted[target_column]
        X = self.prepare_features(df_sorted)
        
        # Split temporal para séries temporais
        train_size = int(len(df_sorted) * 0.8)
        y_train, y_test = y[:train_size], y[train_size:]
        X_train, X_test = X[:train_size], X[train_size:]
        
        # Treinar SARIMAX com parâmetros otimizados
        self.model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),  # ARIMA simplificado
            seasonal_order=(0, 0, 0, 0),  # Sem sazonalidade para acelerar
            trend='c'  # Com constante
        )
        
        # Fit do modelo
        self.fitted_model = self.model.fit(
            disp=False, 
            maxiter=50,
            method='lbfgs'  # Método mais rápido
        )
        
        # Avaliar modelo
        metrics = self._evaluate_sarimax(X_test, y_test)
        self.model_metrics = metrics
        
        logger.info(f"Modelo SARIMAX treinado com MAE: {metrics['mae']:.4f}")
        return metrics
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Faz predições de temperatura"""
        try:
            if self.model is None:
                raise ValueError("Modelo não foi treinado")
            
            if self.model_type == 'sarimax':
                return self._predict_sarimax(df, steps)
            else:
                return self._predict_random_forest(df)
                
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise
    
    def _predict_random_forest(self, df: pd.DataFrame) -> Tuple[np.ndarray, None]:
        """Predições com Random Forest"""
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        return predictions, None
    
    def _predict_sarimax(self, df: pd.DataFrame, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predições com SARIMAX"""
        X = self.prepare_features(df)
        
        # Forecast
        forecast = self.fitted_model.get_forecast(steps=steps, exog=X[:steps])
        predictions = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        
        return predictions.values, confidence_intervals.values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features (apenas Random Forest)"""
        try:
            if self.model_type != 'random_forest' or self.model is None:
                raise ValueError("Feature importance apenas disponível para Random Forest treinado")
            
            # Obter importância do Random Forest
            importance = self.model.named_steps['regressor'].feature_importances_
            
            # Criar DataFrame com importância
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Erro ao obter importância: {e}")
            raise
    
    def save_model(self, model_path: Optional[str] = None):
        """Salva modelo treinado"""
        try:
            if model_path is None:
                model_path = f"models/temperature_predictor_{self.model_type}.pkl"
            
            model_data = {
                'model': self.fitted_model if self.model_type == 'sarimax' else self.model,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'metrics': self.model_metrics,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
            
            # Salvar modelo localmente
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Salvar no MinIO se disponível
            try:
                with open(model_path, 'rb') as f:
                    minio_connection.client.put_object(
                        minio_connection.bucket_name,
                        f"models/temperature_predictor_{self.model_type}.pkl",
                        f,
                        length=-1,
                        part_size=10*1024*1024
                    )
                logger.info("Modelo salvo também no MinIO")
            except Exception as e:
                logger.warning(f"Erro ao salvar no MinIO: {e}")
            
            logger.info(f"Modelo salvo em {model_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None):
        """Carrega modelo salvo"""
        try:
            if model_path is None:
                model_path = f"models/temperature_predictor_{self.model_type}.pkl"
            
            # Tentar carregar do MinIO primeiro
            try:
                response = minio_connection.client.get_object(
                    minio_connection.bucket_name,
                    f"models/temperature_predictor_{self.model_type}.pkl"
                )
                model_data = pickle.load(response)
                
            except Exception:
                # Se não conseguir do MinIO, carregar localmente
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Carregar componentes do modelo
            if self.model_type == 'sarimax':
                self.fitted_model = model_data['model']
            else:
                self.model = model_data['model']
                
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['metrics']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.label_encoder = model_data.get('label_encoder', LabelEncoder())
            
            logger.info(f"Modelo carregado: {model_path}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _evaluate_random_forest(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Avalia performance do modelo Random Forest"""
        try:
            # Predições
            y_pred = self.model.predict(X_test)
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_test, y_test, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mae_mean': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std(),
                'model_type': 'random_forest'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação Random Forest: {e}")
            raise
    
    def _evaluate_sarimax(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Avalia performance do modelo SARIMAX"""
        try:
            # Predições
            forecast_steps = len(y_test)
            forecast = self.fitted_model.get_forecast(steps=forecast_steps, exog=X_test)
            y_pred = forecast.predicted_mean
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # AIC/BIC do modelo
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'aic': aic,
                'bic': bic,
                'model_type': 'sarimax'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação SARIMAX: {e}")
            raise