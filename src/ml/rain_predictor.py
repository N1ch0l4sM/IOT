"""
Modelos de Machine Learning para previsão de chuva
"""
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from src.config import MODEL_CONFIG
from src.utils.database import db_connection
from src.utils.minio_client import minio_connection

logger = logging.getLogger(__name__)

class RainPredictor:
    """Classe para treinamento e predição de chuva"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.model_metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para o modelo"""
        try:
            logger.info("Preparando features para o modelo")
            
            # Selecionar features relevantes
            feature_columns = [
                'temperature', 'humidity', 'pressure', 'wind_speed',
                'feels_like', 'dew_point', 'pressure_tendency',
                'temp_humidity_interaction', 'wind_pressure_interaction',
                'hour', 'day_of_week', 'month'
            ]
            
            # Filtrar apenas colunas que existem no DataFrame
            available_columns = [col for col in feature_columns if col in df.columns]
            
            # Adicionar encoding para wind_direction se existir
            if 'wind_direction' in df.columns:
                df['wind_direction_encoded'] = self.label_encoder.fit_transform(df['wind_direction'])
                available_columns.append('wind_direction_encoded')
            
            self.feature_columns = available_columns
            
            # Retornar apenas as features selecionadas
            return df[available_columns]
            
        except Exception as e:
            logger.error(f"Erro ao preparar features: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'rain_probability') -> Dict:
        """Treina o modelo de previsão"""
        try:
            logger.info("Iniciando treinamento do modelo")
            
            # Preparar features
            X = self.prepare_features(df)
            y = df[target_column]
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Criar pipeline com normalização e modelo
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Treinar modelo
            self.model.fit(X_train, y_train)
            
            # Avaliar modelo
            metrics = self._evaluate_model(X_test, y_test)
            
            # Salvar métricas
            self.model_metrics = metrics
            
            logger.info(f"Modelo treinado com AUC: {metrics['auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Faz predições"""
        try:
            if self.model is None:
                raise ValueError("Modelo não foi treinado")
            
            # Preparar features
            X = self.prepare_features(df)
            
            # Fazer predições
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna importância das features"""
        try:
            if self.model is None:
                raise ValueError("Modelo não foi treinado")
            
            # Obter importância do Random Forest
            importance = self.model.named_steps['classifier'].feature_importances_
            
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
                model_path = f"{MODEL_CONFIG['path']}{MODEL_CONFIG['name']}.pkl"
            
            # Salvar modelo localmente
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'metrics': self.model_metrics,
                    'version': MODEL_CONFIG['version']
                }, f)
            
            # Salvar no MinIO
            with open(model_path, 'rb') as f:
                minio_connection.client.put_object(
                    minio_connection.bucket_name,
                    f"models/{MODEL_CONFIG['name']}.pkl",
                    f,
                    length=-1,
                    part_size=10*1024*1024
                )
            
            logger.info(f"Modelo salvo em {model_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None):
        """Carrega modelo salvo"""
        try:
            if model_path is None:
                model_path = f"{MODEL_CONFIG['path']}{MODEL_CONFIG['name']}.pkl"
            
            # Tentar carregar do MinIO primeiro
            try:
                response = minio_connection.client.get_object(
                    minio_connection.bucket_name,
                    f"models/{MODEL_CONFIG['name']}.pkl"
                )
                model_data = pickle.load(response)
                
            except Exception:
                # Se não conseguir do MinIO, carregar localmente
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Carregar componentes do modelo
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['metrics']
            
            logger.info(f"Modelo carregado: {model_path}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Avalia performance do modelo"""
        try:
            # Predições
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Métricas
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='roc_auc')
            
            # Relatório de classificação
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Matriz de confusão
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            metrics = {
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'precision': class_report['1']['precision'],
                'recall': class_report['1']['recall'],
                'f1_score': class_report['1']['f1-score'],
                'accuracy': class_report['accuracy'],
                'confusion_matrix': conf_matrix.tolist()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {e}")
            raise

def retrain_model():
    """Função para retreinar o modelo periodicamente"""
    try:
        logger.info("Iniciando retreinamento do modelo")
        
        # Carregar dados processados
        query = """
        SELECT * FROM weather_data 
        WHERE recorded_at >= NOW() - INTERVAL '30 days'
        """
        df = db_connection.execute_query(query)
        
        if len(df) < 100:
            logger.warning("Poucos dados disponíveis para retreinamento")
            return
        
        # Criar instância do preditor
        predictor = RainPredictor()
        
        # Treinar modelo
        metrics = predictor.train_model(df)
        
        # Salvar modelo
        predictor.save_model()
        
        # Salvar métricas no banco
        metrics_df = pd.DataFrame([{
            'model_version': MODEL_CONFIG['version'],
            'auc': metrics['auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'trained_at': pd.Timestamp.now()
        }])
        
        db_connection.insert_dataframe(metrics_df, 'model_metrics')
        
        logger.info("Retreinamento concluído")
        
    except Exception as e:
        logger.error(f"Erro no retreinamento: {e}")
        raise
