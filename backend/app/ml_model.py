import numpy as np
import json
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class WallDetectionModel:
    def __init__(self, model_dir="ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Загружаем модель если она существует
        self.load_model()
    
    def extract_features(self, markup_data):
        """Извлечение признаков из разметки"""
        features = []
        
        if 'objects' not in markup_data:
            return np.array([])
        
        for obj in markup_data['objects']:
            if obj['type'] == 'wall':
                # Признаки для стены
                wall_features = []
                
                # Геометрические признаки
                if obj['points']:
                    # Координаты точек
                    points = np.array([[p['x'], p['y']] for p in obj['points']])
                    
                    # Длина стены
                    if len(points) >= 2:
                        lengths = []
                        for i in range(len(points) - 1):
                            length = np.linalg.norm(points[i+1] - points[i])
                            lengths.append(length)
                        
                        wall_features.extend([
                            np.mean(lengths) if lengths else 0,
                            np.std(lengths) if len(lengths) > 1 else 0,
                            np.max(lengths) if lengths else 0,
                            np.min(lengths) if lengths else 0,
                            len(points)  # Количество точек
                        ])
                    
                    # Углы между сегментами
                    if len(points) >= 3:
                        angles = []
                        for i in range(1, len(points) - 1):
                            v1 = points[i] - points[i-1]
                            v2 = points[i+1] - points[i]
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                            angles.append(angle)
                        
                        if angles:
                            wall_features.extend([
                                np.mean(angles),
                                np.std(angles)
                            ])
                    
                    features.append(wall_features)
        
        return np.array(features) if features else np.array([])
    
    def prepare_training_data(self, markups):
        """Подготовка данных для обучения"""
        X = []
        y = []
        
        for markup in markups:
            markup_data = markup['markup_data']
            features = self.extract_features(markup_data)
            
            if len(features) > 0:
                X.extend(features)
                # Для каждой стены метка 1 (это стена)
                y.extend([1] * len(features))
            
            # Также добавляем отрицательные примеры (не стены)
            # Для простоты создаем случайные "не стены"
            if len(features) > 0:
                num_negative = min(len(features), 5)
                for _ in range(num_negative):
                    # Случайные признаки для "не стены"
                    negative_features = np.random.randn(features.shape[1]) * 100
                    X.append(negative_features)
                    y.append(0)
        
        return np.array(X), np.array(y)
    
    def train(self, markups):
        """Обучение модели на размеченных данных"""
        if not markups:
            return False
        
        X, y = self.prepare_training_data(markups)
        
        if len(X) == 0 or len(y) == 0:
            return False
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение модели
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Сохранение модели
        self.save_model()
        
        # Расчет точности
        accuracy = self.model.score(X_scaled, y)
        
        return {
            'accuracy': float(accuracy),
            'samples': len(X),
            'walls_count': np.sum(y == 1),
            'non_walls_count': np.sum(y == 0)
        }
    
    def predict_walls(self, markup_data):
        """Предсказание стен в новой разметке"""
        if not self.is_trained:
            return []
        
        features = self.extract_features(markup_data)
        
        if len(features) == 0:
            return []
        
        # Масштабирование и предсказание
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:  # Это стена
                results.append({
                    'index': i,
                    'is_wall': True,
                    'confidence': float(prob[1]),  # Вероятность что это стена
                    'features': features[i].tolist()
                })
        
        return results
    
    def save_model(self):
        """Сохранение модели на диск"""
        model_path = self.model_dir / "wall_detection_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Сохранение метаданных
        metadata = {
            'is_trained': self.is_trained,
            'model_type': 'RandomForest',
            'saved_at': str(np.datetime64('now'))
        }
        
        with open(self.model_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self):
        """Загрузка модели с диска"""
        model_path = self.model_dir / "wall_detection_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "model_metadata.json"
        
        if model_path.exists() and scaler_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.is_trained = metadata.get('is_trained', False)
                
                return True
            except:
                pass
        
        return False

# Глобальный экземпляр модели
wall_model = WallDetectionModel()