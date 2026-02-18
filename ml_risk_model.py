import pandas as pd
import numpy as np
import re
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sentence_transformers import SentenceTransformer

class ProjectRiskModel:
    def __init__(self, force_retrain=False, use_synthetic=False):
        self.best_model_file = 'best_model.pkl'
        self.use_synthetic = use_synthetic
        
        # Загружаем эмбеддер (многоязычная лёгкая модель)
        print("📚 Загрузка модели эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2)...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Модель эмбеддингов загружена")
        
        if not force_retrain and os.path.exists(self.best_model_file):
            self.load_model()
        else:
            self.train_model()
    
    def preprocess_text(self, text):
        """Минимальная очистка — убираем спецсимволы, оставляем буквы и цифры"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_features(self, projects_data):
        """Извлекает признаки: эмбеддинг текста + числовые характеристики"""
        features = []
        for p in projects_data:
            title = p.get('title', '')
            desc = p.get('description', '')
            # Объединяем заголовок и описание для эмбеддинга
            full_text = f"{title}. {desc}"
            # Очищаем текст (но эмбеддер сам умеет обрабатывать сырой текст, очистка не обязательна)
            clean_text = self.preprocess_text(full_text)
            
            # Получаем эмбеддинг (вектор 384)
            emb = self.embedder.encode(clean_text, show_progress_bar=False)
            
            goal = p.get('goal_amount', 0)
            features.append({
                'embedding': emb,               # numpy array (384,)
                'goal_amount': goal,
                'text_length': len(clean_text),
                'word_count': len(clean_text.split()),
                'goal_category': self._categorize_goal(goal)
            })
        return features
    
    def _categorize_goal(self, amount):
        if amount < 5000: return 0
        elif amount < 25000: return 1
        elif amount < 100000: return 2
        else: return 3
    
    def load_all_data(self):
        """Загружает реальные проекты + опционально синтетику"""
        projects = []
        
        # Реальные проекты (Planeta.ru + добавленные вами)
        if os.path.exists('real_projects.json'):
            with open('real_projects.json', 'r', encoding='utf-8') as f:
                projects.extend(json.load(f))
            print(f"📦 Загружено реальных проектов: {len([p for p in projects if 'real_projects.json' in str(p) or True])}")
        
        # Ваши исходные размеченные (если есть)
        if os.path.exists('training_data.json'):
            with open('training_data.json', 'r', encoding='utf-8') as f:
                projects.extend(json.load(f))
            print(f"📦 Загружено проектов из training_data.json")
        
        # Синтетика (по желанию)
        if self.use_synthetic and os.path.exists('synthetic_data.json'):
            with open('synthetic_data.json', 'r', encoding='utf-8') as f:
                projects.extend(json.load(f))
            print(f"📦 Загружено синтетических проектов")
        
        return projects
    
    def train_model(self):
        print("\n🚀 Начало обучения моделей (с эмбеддингами)...")
        data = self.load_all_data()
        
        if len(data) < 5:
            raise ValueError(f"Недостаточно данных для обучения (нужно минимум 5, получено {len(data)})")
        
        # Извлекаем признаки
        features = self.extract_features(data)
        df = pd.DataFrame(features)
        
        # Разворачиваем эмбеддинг в отдельные колонки
        embeddings = np.vstack(df['embedding'].values)  # (n_samples, 384)
        emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        
        # Убираем колонку embedding и объединяем с числовыми
        df = df.drop('embedding', axis=1)
        df = pd.concat([df, df_emb], axis=1)
        
        # Целевая переменная
        y = np.array([p['risk_score'] for p in data])
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )
        
        # Препроцессор: все признаки (эмбеддинги + числовые) масштабируем
        # Список всех колонок, которые участвуют в обучении
        feature_cols = emb_cols + ['goal_amount', 'text_length', 'word_count', 'goal_category']
        
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), feature_cols)
        ])
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # --- RandomForest ---
        print("🌲 Обучение RandomForest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=7,          # Ограничиваем глубину, чтобы избежать переобучения
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(X_train_processed, y_train)
        pred_rf = rf.predict(X_test_processed)
        rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
        print(f"   RMSE RandomForest: {rmse_rf:.2f}")
        
        # --- GradientBoosting ---
        print("📈 Обучение GradientBoosting...")
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_processed, y_train)
        pred_gb = gb.predict(X_test_processed)
        rmse_gb = np.sqrt(mean_squared_error(y_test, pred_gb))
        print(f"   RMSE GradientBoosting: {rmse_gb:.2f}")
        
        # Выбор лучшей модели
        if rmse_rf <= rmse_gb:
            best_model = rf
            best_name = "RandomForest"
            best_rmse = rmse_rf
        else:
            best_model = gb
            best_name = "GradientBoosting"
            best_rmse = rmse_gb
        
        print(f"🏆 Лучшая модель: {best_name} (RMSE: {best_rmse:.2f})")
        
        # Сохраняем полный пайплайн
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', best_model)
        ])
        joblib.dump(self.pipeline, self.best_model_file)
        print(f"💾 Модель сохранена в {self.best_model_file}")
    
    def predict_risk_score(self, project_data):
        """Предсказание риск-скора для одного проекта"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            if not self.load_model():
                print("⚠️ Модель не загружена, возвращаем риск = 50")
                return 50
        
        # Извлекаем признаки так же, как при обучении
        features = self.extract_features([project_data])
        df = pd.DataFrame(features)
        
        # Разворачиваем эмбеддинг
        embeddings = np.vstack(df['embedding'].values)
        emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        df = df.drop('embedding', axis=1)
        df = pd.concat([df, df_emb], axis=1)
        
        # Убеждаемся, что колонки в том же порядке, что и при обучении
        # (можно просто передать все, pipeline сам возьмёт нужные)
        prediction = self.pipeline.predict(df)[0]
        # Ограничиваем диапазон 0-100 и округляем
        return round(max(0, min(100, prediction)), 1)
    
    def load_model(self):
        if os.path.exists(self.best_model_file):
            self.pipeline = joblib.load(self.best_model_file)
            print(f"📂 Загружена лучшая модель из {self.best_model_file}")
            return True
        return False

# Глобальный экземпляр для app.py
risk_model = ProjectRiskModel(force_retrain=True)  # force_retrain=True, чтобы сразу переобучить