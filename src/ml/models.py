import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor, 
                             GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier, MLPRegressor
from typing import Dict, List, Tuple, Any
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ProductRecommendationML:    
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        estimators = []
        
        rf_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=5, 
            min_samples_leaf=3, class_weight='balanced', random_state=42
        )
        estimators.append(('rf', rf_classifier))
        
        gb_classifier = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            subsample=0.8, min_samples_split=5, random_state=42
        )
        estimators.append(('gb', gb_classifier))
        
        if XGBOOST_AVAILABLE:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
            estimators.append(('xgb', xgb_classifier))
            
        if LIGHTGBM_AVAILABLE:
            lgb_classifier = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
            estimators.append(('lgb', lgb_classifier))
        
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=0.001, max_iter=500, random_state=42
        )
        estimators.append(('mlp', mlp_classifier))
        
        self.product_classifier = VotingClassifier(
            estimators=estimators, voting='soft'
        )
        
        benefit_estimators = []
        
        gb_regressor = GradientBoostingRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_split=5, alpha=0.9, random_state=42
        )
        benefit_estimators.append(('gb', gb_regressor))
        
        if XGBOOST_AVAILABLE:
            xgb_regressor = xgb.XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
            benefit_estimators.append(('xgb', xgb_regressor))
            
        if LIGHTGBM_AVAILABLE:
            lgb_regressor = lgb.LGBMRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
            benefit_estimators.append(('lgb', lgb_regressor))
        
        mlp_regressor = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=0.001, max_iter=500, random_state=42
        )
        benefit_estimators.append(('mlp', mlp_regressor))
        
        from sklearn.ensemble import VotingRegressor
        self.benefit_regressor = VotingRegressor(estimators=benefit_estimators)
        
        self.customer_segmentation = KMeans(n_clusters=6, random_state=42, n_init=20)
        self.behavioral_clustering = DBSCAN(eps=0.5, min_samples=3)
        
        self.is_trained = False
        self.feature_importance_aggregated = None
        
    def train_product_classifier(self, X: np.ndarray, y: List[str]) -> Dict[str, Any]:
        unique_labels, counts = np.unique(y, return_counts=True)
        can_stratify = all(count >= 2 for count in counts) and len(X) > 10
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        
        if len(X_train) > 20:
            X_train_poly = self.poly_features.fit_transform(X_train)
            X_test_poly = self.poly_features.transform(X_test)
            X_full_poly = self.poly_features.transform(X)
        else:
            X_train_poly, X_test_poly, X_full_poly = X_train, X_test, X
        
        self.product_classifier.fit(X_train_poly, y_train)
        
        train_score = self.product_classifier.score(X_train_poly, y_train)
        test_score = self.product_classifier.score(X_test_poly, y_test)
        
        try:
            cv_folds = min(3, len(unique_labels))
            cv_scores = cross_val_score(self.product_classifier, X_full_poly, y, cv=cv_folds)
        except Exception:
            cv_scores = np.array([test_score])
        
        y_pred = self.product_classifier.predict(X_test_poly)
        y_pred_proba = self.product_classifier.predict_proba(X_test_poly)
        
        self._aggregate_feature_importance(X_train_poly.shape[1])
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self.feature_importance_aggregated,
            'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
    
    def _aggregate_feature_importance(self, n_features: int):
        importances = []
        
        try:
            if hasattr(self.product_classifier.named_estimators_['rf'], 'feature_importances_'):
                importances.append(self.product_classifier.named_estimators_['rf'].feature_importances_)
        except: pass
        
        try:
            if hasattr(self.product_classifier.named_estimators_['gb'], 'feature_importances_'):
                importances.append(self.product_classifier.named_estimators_['gb'].feature_importances_)
        except: pass
        
        try:
            if hasattr(self.product_classifier.named_estimators_['xgb'], 'feature_importances_'):
                importances.append(self.product_classifier.named_estimators_['xgb'].feature_importances_)
        except: pass
        
        try:
            if hasattr(self.product_classifier.named_estimators_['lgb'], 'feature_importances_'):
                importances.append(self.product_classifier.named_estimators_['lgb'].feature_importances_)
        except: pass
        
        if importances:
            self.feature_importance_aggregated = np.mean(importances, axis=0)
        else:
            self.feature_importance_aggregated = np.ones(n_features) / n_features
    
    def train_benefit_regressor(self, X: np.ndarray, benefits: List[float]) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, benefits, test_size=0.25, random_state=42
        )
        
        if len(X_train) > 20:
            X_train_poly = self.poly_features.transform(X_train)
            X_test_poly = self.poly_features.transform(X_test)
        else:
            X_train_poly, X_test_poly = X_train, X_test
        
        self.benefit_regressor.fit(X_train_poly, y_train)
        
        train_pred = self.benefit_regressor.predict(X_train_poly)
        test_pred = self.benefit_regressor.predict(X_test_poly)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        from sklearn.metrics import r2_score, mean_absolute_error
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'prediction_std': np.std(test_pred)
        }
    
    def segment_customers(self, X: np.ndarray) -> Dict[str, Any]:
        kmeans_clusters = self.customer_segmentation.fit_predict(X)
        
        try:
            behavioral_clusters = self.behavioral_clustering.fit_predict(X)
            behavioral_silhouette = silhouette_score(X, behavioral_clusters) if len(np.unique(behavioral_clusters)) > 1 else -1
        except:
            behavioral_clusters = np.zeros(len(X))
            behavioral_silhouette = -1
        
        kmeans_silhouette = silhouette_score(X, kmeans_clusters)
        
        cluster_analysis = {}
        for i in range(self.customer_segmentation.n_clusters):
            cluster_mask = kmeans_clusters == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                cluster_center = self.customer_segmentation.cluster_centers_[i]
                cluster_data = X[cluster_mask]
                
                cluster_analysis[f'cluster_{i}'] = {
                    'size': int(cluster_size),
                    'percentage': float(cluster_size / len(kmeans_clusters) * 100),
                    'center': cluster_center.tolist(),
                    'variance': float(np.mean(np.var(cluster_data, axis=0))),
                    'cohesion': float(np.mean(np.linalg.norm(cluster_data - cluster_center, axis=1)))
                }
        
        behavioral_analysis = {}
        unique_behavioral = np.unique(behavioral_clusters)
        for i in unique_behavioral:
            if i != -1:
                cluster_mask = behavioral_clusters == i
                cluster_size = np.sum(cluster_mask)
                behavioral_analysis[f'behavioral_{i}'] = {
                    'size': int(cluster_size),
                    'percentage': float(cluster_size / len(behavioral_clusters) * 100)
                }
        
        return {
            'kmeans_clusters': kmeans_clusters,
            'behavioral_clusters': behavioral_clusters,
            'analysis': cluster_analysis,
            'behavioral_analysis': behavioral_analysis,
            'kmeans_inertia': self.customer_segmentation.inertia_,
            'kmeans_silhouette': kmeans_silhouette,
            'behavioral_silhouette': behavioral_silhouette,
            'optimal_clusters': self._find_optimal_clusters(X)
        }
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        if len(X) < 10:
            return 3
            
        inertias = []
        silhouettes = []
        K_range = range(2, min(8, len(X)//2))
        
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, clusters))
            except:
                continue
        
        if silhouettes:
            optimal_k = K_range[np.argmax(silhouettes)]
            return optimal_k
        else:
            return 3
    
    def predict_product(self, client_features: np.ndarray) -> Tuple[str, float, float]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        features_reshaped = client_features.reshape(1, -1)
        
        try:
            features_poly = self.poly_features.transform(features_reshaped)
        except:
            features_poly = features_reshaped
            
        probabilities = self.product_classifier.predict_proba(features_poly)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        predicted_product = self.product_classifier.classes_[predicted_class_idx]
        predicted_benefit = self.benefit_regressor.predict(features_poly)[0]
        
        uncertainty = 1 - confidence
        adjusted_benefit = predicted_benefit * (1 - uncertainty * 0.3)
        
        return predicted_product, confidence, adjusted_benefit
    
    def predict_top_products(self, client_features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, float]]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        features_reshaped = client_features.reshape(1, -1)
        
        try:
            features_poly = self.poly_features.transform(features_reshaped)
        except:
            features_poly = features_reshaped
            
        probabilities = self.product_classifier.predict_proba(features_poly)[0]
        predicted_benefit = self.benefit_regressor.predict(features_poly)[0]
        
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            product = self.product_classifier.classes_[idx]
            confidence = probabilities[idx]
            uncertainty = 1 - confidence
            adjusted_benefit = predicted_benefit * (1 - uncertainty * 0.3) * confidence
            results.append((product, confidence, adjusted_benefit))
        
        return results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if not self.is_trained:
            return {}
            
        importance_dict = {}
        
        classifier_importance = dict(zip(
            feature_names, 
            self.product_classifier.named_estimators_['rf'].feature_importances_
        ))
        
        try:
            regressor_importance = dict(zip(
                feature_names,
                self.benefit_regressor.named_estimators_['gb'].feature_importances_
            ))
        except:
            regressor_importance = {f: 0.1 for f in feature_names}
        
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
        best_hour = 19 
        best_probability = 0.5
        
        hour_scores = {}
        for hour in range(7, 24):
            for day in [0, 5]:  
                features = self.create_timing_features(hour, day, client_features)
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
