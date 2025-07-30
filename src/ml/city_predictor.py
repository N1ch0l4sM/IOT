import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging

class CityTemperaturePredictor:
    """
    Preditor de temperatura por cidade usando SARIMAX.
    Usa apenas valores passados (com defasagem) para previsões.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (2, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
                 exog_lags: int = 1):
        """
        Inicializar preditor de temperatura por cidade.
        
        Args:
            order: Ordem ARIMA (p, d, q)
            seasonal_order: Ordem sazonal (P, D, Q, s)
            exog_lags: Quantas horas de defasagem usar para variáveis exógenas
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_lags = exog_lags
        self.models: Dict[str, SARIMAX] = {}
        self.fitted_models: Dict[str, any] = {}
        self.fitted_models_no_exog: Dict[str, any] = {}
        self.city_data: Dict[str, pd.DataFrame] = {}
        self.training_data: Dict[str, Dict] = {}  # Guardar dados de treino/teste
        self.exog_columns: List[str] = []
        self.lagged_exog_cols: List[str] = []
        self.logger = logging.getLogger(__name__)
        
    def create_lagged_features(self, df: pd.DataFrame, 
                              exog_columns: List[str] = None) -> pd.DataFrame:
        """
        Criar features com defasagem temporal para variáveis exógenas.
        
        Args:
            df: DataFrame com dados temporais ordenados por data
            exog_columns: Colunas para criar defasagem
            
        Returns:
            DataFrame com features defasadas
        """
        if exog_columns is None:
            return df
            
        df_lagged = df.copy()
        
        for col in exog_columns:
            if col in df.columns:
                for lag in range(1, self.exog_lags + 1):
                    lag_col_name = f"{col}_lag_{lag}"
                    df_lagged[lag_col_name] = df[col].shift(lag)
                    
                df_lagged[f"{col}_ma_3"] = df[col].shift(1).rolling(window=3).mean()
                df_lagged[f"{col}_ma_6"] = df[col].shift(1).rolling(window=6).mean()
        
        return df_lagged
    
    def prepare_city_data(self, df: pd.DataFrame, 
                         city_column: str = 'city',
                         exog_columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Preparar dados por cidade com features defasadas.
        
        Args:
            df: DataFrame com dados
            city_column: Coluna de identificação da cidade  
            exog_columns: Colunas para variáveis exógenas
            
        Returns:
            Dict com dados preparados por cidade
        """
        city_data = {}
        self.exog_columns = exog_columns or []

        for city in df[city_column].unique():
            city_df = df[df[city_column] == city].copy()
            
            city_df = city_df.sort_values('date')
            
            if len(city_df) < 200:
                self.logger.warning(f"Cidade {city} tem poucos dados ({len(city_df)})")
                continue
                
            city_df.set_index('date', inplace=True)
            city_df = self.create_lagged_features(city_df, exog_columns)
            city_df = city_df.dropna()
            
            if len(city_df) < 100:
                self.logger.warning(f"Cidade {city} tem poucos dados após defasagens")
                continue
                
            city_data[city] = city_df
            
        return city_data
    
    def get_lagged_exog_columns(self, exog_columns: List[str]) -> List[str]:
        """
        Obter nomes das colunas com defasagem.
        
        Args:
            exog_columns: Lista de colunas originais
            
        Returns:
            Lista de colunas com defasagem
        """
        if not exog_columns:
            return []
            
        lagged_cols = []
        for col in exog_columns:
            for lag in range(1, self.exog_lags + 1):
                lagged_cols.append(f"{col}_lag_{lag}")
            lagged_cols.extend([f"{col}_ma_3", f"{col}_ma_6"])
            
        return lagged_cols
    
    def create_future_exog(self, city_df: pd.DataFrame, steps: int) -> Optional[pd.DataFrame]:
        """
        Criar variáveis exógenas para previsão futura usando valores mais recentes.
        
        Args:
            city_df: DataFrame da cidade com dados históricos
            steps: Número de passos futuros
            
        Returns:
            DataFrame com exógenas futuras ou None
        """
        if not self.lagged_exog_cols:
            return None
            
        last_values = {}
        for col in self.exog_columns:
            if col in city_df.columns:
                last_values[col] = city_df[col].tail(max(self.exog_lags, 6)).values
        
        if not last_values:
            return None
        
        future_exog_data = []
        
        for step in range(steps):
            step_data = {}
            
            for col in self.exog_columns:
                if col not in last_values:
                    continue
                    
                col_values = last_values[col]
                
                for lag in range(1, self.exog_lags + 1):
                    lag_col = f"{col}_lag_{lag}"
                    if lag <= len(col_values):
                        step_data[lag_col] = col_values[-(lag + step)] if (lag + step) <= len(col_values) else col_values[-1]
                    else:
                        step_data[lag_col] = col_values[-1]
                
                if len(col_values) >= 3:
                    step_data[f"{col}_ma_3"] = np.mean(col_values[-3:])
                if len(col_values) >= 6:
                    step_data[f"{col}_ma_6"] = np.mean(col_values[-6:])
            
            future_exog_data.append(step_data)
        
        available_cols = [col for col in self.lagged_exog_cols if col in future_exog_data[0].keys()]
        
        if not available_cols:
            return None
            
        future_exog_df = pd.DataFrame(future_exog_data, columns=available_cols)
        future_exog_df = future_exog_df.fillna(method='ffill').fillna(method='bfill')
        
        return future_exog_df[available_cols] if len(available_cols) > 0 else None
    
    def train_city_models(self, df: pd.DataFrame, 
                         target_column: str = 'temperature',
                         city_column: str = 'city', 
                         exog_columns: List[str] = None,
                         test_size: float = 0.2) -> List[str]:
        """
        Treinar modelos SARIMAX por cidade.
        Função focada apenas no treinamento.
        
        Args:
            df: DataFrame com dados
            target_column: Coluna alvo
            city_column: Coluna de cidade
            exog_columns: Variáveis exógenas (serão defasadas)
            test_size: Proporção para teste
            
        Returns:
            Lista de cidades treinadas com sucesso
        """
        self.logger.info("Iniciando treinamento de modelos SARIMAX por cidade")
        
        # Preparar dados
        city_data = self.prepare_city_data(df, city_column, exog_columns)
        self.city_data = city_data
        self.lagged_exog_cols = self.get_lagged_exog_columns(exog_columns)
        
        trained_cities = []
        
        for city, city_df in city_data.items():
            try:
                self.logger.info(f"Treinando modelo para cidade: {city}")
                
                # Preparar dados
                y = city_df[target_column]
                exog = None
                available_cols = []
                
                if self.lagged_exog_cols:
                    available_cols = [col for col in self.lagged_exog_cols if col in city_df.columns]
                    if available_cols:
                        exog = city_df[available_cols]
                
                # Divisão treino/teste
                train_size = int((1 - test_size) * len(y))
                y_train = y[:train_size]
                y_test = y[train_size:]
                
                exog_train = None
                exog_test = None
                if exog is not None:
                    exog_train = exog[:train_size]
                    exog_test = exog[train_size:]
                
                # Treinar modelo com exógenas
                model_with_exog = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model_with_exog = model_with_exog.fit(disp=False, maxiter=100)
                
                # Treinar modelo sem exógenas (backup)
                model_no_exog = SARIMAX(
                    y_train,
                    exog=None,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model_no_exog = model_no_exog.fit(disp=False, maxiter=100)
                
                # Armazenar modelos e dados de treino/teste
                self.fitted_models[city] = fitted_model_with_exog
                self.fitted_models_no_exog[city] = fitted_model_no_exog
                
                # Guardar dados para análise posterior
                self.training_data[city] = {
                    'y_train': y_train,
                    'y_test': y_test,
                    'exog_train': exog_train,
                    'exog_test': exog_test,
                    'available_cols': available_cols,
                    'train_size': len(y_train),
                    'test_size': len(y_test)
                }
                
                trained_cities.append(city)
                self.logger.info(f"Modelo treinado com sucesso para cidade: {city}")
                
            except Exception as e:
                self.logger.error(f"Erro ao treinar cidade {city}: {str(e)}")
                continue
        
        self.logger.info(f"Treinamento concluído. {len(trained_cities)} cidades treinadas.")
        return trained_cities
    
    def evaluate_models(self) -> Dict[str, Dict]:
        """
        Avaliar performance dos modelos treinados.
        Função dedicada à análise de métricas.
        
        Returns:
            Dict com métricas por cidade
        """
        if not self.fitted_models:
            raise ValueError("Nenhum modelo treinado. Execute train_city_models() primeiro.")
        
        self.logger.info("Iniciando avaliação de modelos")
        
        all_metrics = {}
        
        for city in self.fitted_models.keys():
            try:
                # Recuperar dados de treino/teste
                train_data = self.training_data[city]
                y_test = train_data['y_test']
                exog_test = train_data['exog_test']
                
                model = self.fitted_models[city]
                model_no_exog = self.fitted_models_no_exog[city]
                
                # Fazer predições
                if exog_test is not None:
                    predictions = model.forecast(steps=len(y_test), exog=exog_test)
                    model_used = 'with_exog'
                else:
                    predictions = model_no_exog.forecast(steps=len(y_test))
                    model_used = 'no_exog'
                
                # Calcular métricas
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
                
                # Métricas do modelo
                model_metrics = model if exog_test is not None else model_no_exog
                
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'aic': model_metrics.aic,
                    'bic': model_metrics.bic,
                    'train_size': train_data['train_size'],
                    'test_size': train_data['test_size'],
                    'exog_features': len(train_data['available_cols']),
                    'model_used': model_used,
                    'predictions': predictions.tolist(),
                    'actual': y_test.tolist()
                }
                
                all_metrics[city] = metrics
                
                self.logger.info(
                    f"Cidade {city} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, "
                    f"R²: {r2:.4f}, Modelo: {model_used}"
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao avaliar cidade {city}: {str(e)}")
                continue
        
        return all_metrics
    
    def get_model_diagnostics(self, city: str) -> Dict[str, any]:
        """
        Obter diagnósticos detalhados de um modelo específico.
        
        Args:
            city: Nome da cidade
            
        Returns:
            Dict com diagnósticos do modelo
        """
        if city not in self.fitted_models:
            raise ValueError(f"Modelo não encontrado para cidade: {city}")
        
        model = self.fitted_models[city]
        
        try:
            diagnostics = {
                'summary': str(model.summary()),
                'aic': model.aic,
                'bic': model.bic,
                'llf': model.llf,
                'params': model.params.to_dict(),
                'pvalues': model.pvalues.to_dict(),
                'residuals_stats': {
                    'mean': np.mean(model.resid),
                    'std': np.std(model.resid),
                    'skew': float(model.resid.skew()) if hasattr(model.resid, 'skew') else None,
                    'kurtosis': float(model.resid.kurtosis()) if hasattr(model.resid, 'kurtosis') else None
                }
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Erro ao obter diagnósticos para cidade {city}: {str(e)}")
            return {}
    
    def predict_future(self, city: str, steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fazer previsões futuras usando dados passados mais recentes.
        
        Args:
            city: Nome da cidade
            steps: Passos futuros para prever
            
        Returns:
            Tuple (previsões, intervalos de confiança)
        """
        if city not in self.fitted_models:
            raise ValueError(f"Modelo não encontrado para cidade: {city}")
        
        if city not in self.city_data:
            raise ValueError(f"Dados históricos não encontrados para cidade: {city}")
        
        city_df = self.city_data[city]
        model = self.fitted_models[city]
        
        try:
            future_exog = self.create_future_exog(city_df, steps)
            
            if future_exog is not None and len(future_exog.columns) > 0:
                self.logger.info(f"Usando {len(future_exog.columns)} variáveis exógenas para previsão")
                forecast_result = model.get_forecast(steps=steps, exog=future_exog)
            else:
                self.logger.info("Usando apenas componente temporal para previsão")
                model_no_exog = self.fitted_models_no_exog[city]
                forecast_result = model_no_exog.get_forecast(steps=steps)
            
            predictions = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            return predictions.values, conf_int.values
            
        except Exception as e:
            self.logger.warning(f"Erro com exógenas, usando modelo temporal: {str(e)}")
            model_no_exog = self.fitted_models_no_exog[city]
            forecast_result = model_no_exog.get_forecast(steps=steps)
            predictions = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            return predictions.values, conf_int.values
    
    def save_models(self, base_path: str = "models/temperature_by_city") -> None:
        """
        Salvar todos os modelos treinados e metadados.
        
        Args:
            base_path: Caminho base para salvar os modelos
        """
        os.makedirs(base_path, exist_ok=True)
        
        # Salvar modelos
        for city, model in self.fitted_models.items():
            city_safe = str(city).replace(" ", "_").replace("/", "_")
            filepath = os.path.join(base_path, f"temp_model_exog_{city_safe}.pkl")
            joblib.dump(model, filepath)
            
        for city, model in self.fitted_models_no_exog.items():
            city_safe = str(city).replace(" ", "_").replace("/", "_")
            filepath = os.path.join(base_path, f"temp_model_no_exog_{city_safe}.pkl")
            joblib.dump(model, filepath)
        
        # Salvar metadados
        metadata = {
            'exog_columns': self.exog_columns,
            'lagged_exog_cols': self.lagged_exog_cols,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'exog_lags': self.exog_lags,
            'training_data': self.training_data
        }
        
        metadata_path = os.path.join(base_path, "model_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"Modelos e metadados salvos em: {base_path}")