import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor, 
                             GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier)
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Any
import joblib

class ProductRecommendationML:    
    def __init__(self):
        rf_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, 
            class_weight='balanced', random_state=42
        )
        gb_classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1, 
            subsample=0.8, random_state=42
        )
        et_classifier = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=42
        )
        lr_classifier = LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        )
        
        # VotingClassifier - ансамбль сильных моделей  
        self.product_classifier = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('gb', gb_classifier), 
                ('et', et_classifier),
                ('lr', lr_classifier)
            ],
            voting='soft'  # используем вероятности для более умных решений
        )
        
        # Более мощный регрессор с гиперпараметрами
        self.benefit_regressor = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        )
        
        # Улучшенная кластеризация
        self.customer_segmentation = KMeans(n_clusters=8, random_state=42, n_init=20)
        self.is_trained = False
        
    def train_product_classifier(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        # Проверяем, можно ли использовать стратификацию
        unique_labels, counts = np.unique(y, return_counts=True)
        can_stratify = all(count >= 2 for count in counts) and len(X) > 10
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        self.product_classifier.fit(X_train, y_train)
        
        train_score = self.product_classifier.score(X_train, y_train)
        test_score = self.product_classifier.score(X_test, y_test)
        
        try:
            cv_folds = min(5, len(unique_labels))  # не больше количества классов
            cv_scores = cross_val_score(self.product_classifier, X, y, cv=cv_folds)
        except Exception:
            cv_scores = np.array([test_score])  
        
        # Предсказания для отчета
        y_pred = self.product_classifier.predict(X_test)
        
        try:
            # Получаем важность от RandomForest компонента
            feature_importance = self.product_classifier.named_estimators_['rf'].feature_importances_
        except:
           
            feature_importance = np.ones(X.shape[1]) / X.shape[1]
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def train_benefit_regressor(self, X: np.ndarray, benefits: List[float]) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, benefits, test_size=0.2, random_state=42
        )
        
        self.benefit_regressor.fit(X_train, y_train)
        
        train_pred = self.benefit_regressor.predict(X_train)
        test_pred = self.benefit_regressor.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'feature_importance': self.benefit_regressor.feature_importances_
        }
    
    def segment_customers(self, X: np.ndarray) -> Dict[str, Any]:
        clusters = self.customer_segmentation.fit_predict(X)
        
        cluster_analysis = {}
        for i in range(self.customer_segmentation.n_clusters):
            cluster_mask = clusters == i
            cluster_size = np.sum(cluster_mask)
            cluster_center = self.customer_segmentation.cluster_centers_[i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(clusters) * 100),
                'center': cluster_center.tolist()
            }
        
        return {
            'clusters': clusters,
            'analysis': cluster_analysis,
            'inertia': self.customer_segmentation.inertia_
        }
    
    def predict_product(self, client_features: np.ndarray) -> Tuple[str, float]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        product_proba = self.product_classifier.predict_proba(client_features.reshape(1, -1))[0]
        product_idx = np.argmax(product_proba)
        product = self.product_classifier.classes_[product_idx]
        confidence = product_proba[product_idx]
        
        benefit = self.benefit_regressor.predict(client_features.reshape(1, -1))[0]
        
        return product, confidence, benefit
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if not self.is_trained:
            return {}
            
        importance_dict = {}
        
        classifier_importance = dict(zip(
            feature_names, 
            self.product_classifier.named_estimators_['rf'].feature_importances_
        ))
        
        regressor_importance = dict(zip(
            feature_names,
            self.benefit_regressor.feature_importances_
        ))
        
        for feature in feature_names:
            combined_importance = (
                classifier_importance.get(feature, 0) + 
                regressor_importance.get(feature, 0)
            ) / 2
            importance_dict[feature] = combined_importance
            
        return importance_dict
    
    def save_model(self, filepath: str):
        model_data = {
            'product_classifier': self.product_classifier,
            'benefit_regressor': self.benefit_regressor,
            'customer_segmentation': self.customer_segmentation,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.product_classifier = model_data['product_classifier']
        self.benefit_regressor = model_data['benefit_regressor']
        self.customer_segmentation = model_data['customer_segmentation']
        self.is_trained = model_data['is_trained']


class TimingOptimizationML:
    
    def __init__(self):
        self.timing_classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.response_regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
    
    def create_timing_features(self, hour: int, day_of_week: int, client_features: Dict) -> np.ndarray: 
        features = [
            hour,
            day_of_week,
            1 if day_of_week >= 5 else 0,  
            1 if 9 <= hour <= 17 else 0,  
            1 if 18 <= hour <= 22 else 0,  
            1 if hour <= 6 or hour >= 23 else 0,  
            client_features.get('age', 30),
            client_features.get('is_student', 0),
            client_features.get('is_premium', 0),
            client_features.get('spending_to_balance_ratio', 0.5)
        ]
        return np.array(features)
    
    def predict_optimal_timing(self, client_features: Dict) -> Dict[str, Any]:
        best_hour = 19  # default
        best_probability = 0.5
        
        hour_scores = {}
        for hour in range(7, 24):
            for day in [0, 5]:  
                features = self.create_timing_features(hour, day, client_features)
                # Здесь была бы предсказанная вероятность отклика
                score = self._calculate_timing_score(hour, day, client_features)
                hour_scores[(hour, day)] = score
        
        best_time = max(hour_scores.items(), key=lambda x: x[1])
        
        return {
            'optimal_hour': best_time[0][0],
            'optimal_day_type': 'weekend' if best_time[0][1] >= 5 else 'weekday',
            'expected_response_rate': best_time[1],
            'all_scores': hour_scores
        }
    
    def _calculate_timing_score(self, hour: int, day_of_week: int, client_features: Dict) -> float:
        score = 0.5
        
        is_weekend = day_of_week >= 5
        
        if client_features.get('is_student', 0):
            if is_weekend:
                if 13 <= hour <= 16 or 19 <= hour <= 23:
                    score += 0.4
            else:
                if 15 <= hour <= 17 or 19 <= hour <= 22:
                    score += 0.4
        
        elif client_features.get('is_premium', 0):
            if 10 <= hour <= 12 or 19 <= hour <= 22:
                score += 0.3
        
        elif client_features.get('is_salary', 0):
            if is_weekend:
                if 11 <= hour <= 15:
                    score += 0.4
            else:
                if 12 <= hour <= 14 or 18 <= hour <= 20:
                    score += 0.4
        
        return min(score, 1.0)
