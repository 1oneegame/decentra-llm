import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime

from src.services.data_loader import DataLoader
from src.models.client import ClientAnalysis
from src.models.products import ProductCatalog
from src.ml.feature_engineering import FeatureEngineer
from src.ml.models import ProductRecommendationML, TimingOptimizationML

from src.services.client_analyzer import ClientAnalyzer
from src.services.push_generator import PushGenerator

class MLPushSystem:
    """ML-Система персонализированных пуш-уведомлений"""
    
    def __init__(self, data_path: str = 'data/raw/dataset'):
        self.data_loader = DataLoader(data_path)
        self.feature_engineer = FeatureEngineer()
        self.ml_model = ProductRecommendationML()
        self.timing_model = TimingOptimizationML()
        self.product_catalog = ProductCatalog.get_all_products()
        
        
        self.client_analyzer = ClientAnalyzer()
        self.push_generator = PushGenerator()
        
        self.clients_data = {}
        self.ml_features = []
        self.training_labels = []
        self.training_benefits = []
        
    def load_and_prepare_data(self):
            
        print("🔄 Загрузка данных клиентов...")
        
        profiles = self.data_loader.load_client_profiles()
                
        print(f"✅ Загружено {len(profiles)} клиентов")
        
        
        print("🔄 Создание ML фичей...")
        
        for client_code in profiles.keys():
            try:
                
                profile = profiles[client_code]
                transactions = self.data_loader.load_client_transactions(client_code)
                transfers = self.data_loader.load_client_transfers(client_code)
                
                
                analysis = self.client_analyzer.analyze_client_behavior(profile, transactions, transfers)
                
                
                basic_features = self.feature_engineer.create_features(analysis)
                behavioral_features = self.client_analyzer.create_behavioral_features(transactions, transfers)
                
                
                all_features = {**basic_features, **behavioral_features}
                self.ml_features.append(all_features)
                
                
                best_product = self._create_behavioral_training_labels(analysis, client_code)
                benefit = self._calculate_benefit_heuristic(analysis, best_product)
                
                self.training_labels.append(best_product)
                self.training_benefits.append(benefit)
                
                self.clients_data[client_code] = {
                    'analysis': analysis,
                    'features': all_features,
                    'transactions': transactions,
                    'transfers': transfers
                }
                
            except Exception as e:
                print(f"❌ Ошибка обработки клиента {client_code}: {e}")
                
                transactions_exist = self.data_loader.load_client_transactions(client_code)
                transfers_exist = self.data_loader.load_client_transfers(client_code)
                print(f"   📊 Данные клиента {client_code}: транзакции={len(transactions_exist)}, переводы={len(transfers_exist)}")
                
                self.ml_features.append({})
                self.training_labels.append('Депозит накопительный')  # По умолчанию
                self.training_benefits.append(0.0)
        
        print(f"✅ Создано {len(self.ml_features)} наборов фичей")
    
    def train_ml_models(self) -> Dict[str, Any]:
        
        print("🤖 Обучение ML моделей...")
        
        if not self.ml_features:
            raise ValueError("Нет данных для обучения! Сначала вызовите load_and_prepare_data()")
        
        
        X, feature_names = self.feature_engineer.prepare_dataset(self.ml_features)
        
        print(f"📊 Размер датасета: {X.shape}")
        print(f"🔢 Количество фичей: {len(feature_names)}")
        print(f"🎯 Уникальных продуктов: {len(set(self.training_labels))}")
        
        
        print("🔄 Обучение классификатора продуктов...")
        classifier_metrics = self.ml_model.train_product_classifier(X, self.training_labels)
        
        
        print("🔄 Обучение регрессора выгоды...")
        regressor_metrics = self.ml_model.train_benefit_regressor(X, self.training_benefits)
        
        
        print("🔄 Сегментация клиентов...")
        segmentation_results = self.ml_model.segment_customers(X)
        
        
        feature_importance = self.ml_model.get_feature_importance(feature_names)
        
        self.ml_model.is_trained = True
        
        # Сохраняем метрики в объекте для доступа через API
        self.training_metrics = {
            'classifier_accuracy': classifier_metrics.get('test_accuracy', 0),
            'regressor_rmse': regressor_metrics.get('test_rmse', 0),
            'clustering_score': segmentation_results.get('kmeans_silhouette', 0),
            'clusters_count': len(segmentation_results.get('analysis', [])),
            'dataset_shape': X.shape,
            'feature_count': len(feature_names)
        }
        
        results = {
            'classifier_metrics': classifier_metrics,
            'regressor_metrics': regressor_metrics,
            'segmentation': segmentation_results,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'dataset_shape': X.shape
        }
        
        print("✅ ML модели обучены!")
        return results
    
    def predict_with_ml(self, client_code: int) -> Dict[str, Any]:
        
        if not self.ml_model.is_trained:
            raise ValueError("ML модель не обучена!")
        
        if client_code not in self.clients_data:
            raise ValueError(f"Клиент {client_code} не найден!")
        
        client_data = self.clients_data[client_code]
        features = client_data['features']
        
        
        features_df = pd.DataFrame([features])
        X = self.feature_engineer.scaler.transform(features_df)
        
        
        ml_product, ml_confidence, ml_benefit = self.ml_model.predict_product(X[0])
        
        
        timing_prediction = self.timing_model.predict_optimal_timing(features)
        
        
        try:
            cluster = self.ml_model.customer_segmentation.predict(X)[0]
        except:
            cluster = 0
        
        
        analysis = client_data['analysis']
        push_message = self.push_generator.generate_push(
            analysis, ml_product, cluster
        )
        
        return {
            'client_code': client_code,
            'ml_prediction': {
                'product': ml_product,
                'confidence': ml_confidence,
                'expected_benefit': ml_benefit,
                'cluster': int(cluster),
                'cluster_description': self.push_generator.get_cluster_description(int(cluster)),
                'push_notification': push_message
            },
            'timing_optimization': timing_prediction,
            'features_used': features
        }
    
    def compare_ml_vs_rules(self) -> Dict[str, Any]:
        
        print("⚔️ Сравнение ML vs Rules...")
        
        ml_predictions = []
        rule_predictions = []
        agreements = 0
        
        for client_code in self.clients_data.keys():
            try:
                
                ml_result = self.predict_with_ml(client_code)
                ml_product = ml_result['ml_prediction']['product']
                
                
                analysis = self.clients_data[client_code]['analysis']
                rule_product = self._create_training_labels_from_spending_patterns(analysis)
                
                ml_predictions.append(ml_product)
                rule_predictions.append(rule_product)
                
                if ml_product == rule_product:
                    agreements += 1
                    
            except Exception as e:
                print(f"❌ Ошибка сравнения для клиента {client_code}: {e}")
        
        agreement_rate = agreements / len(ml_predictions) if ml_predictions else 0
        
        
        comparison_df = pd.DataFrame({
            'ML': ml_predictions,
            'Rules': rule_predictions
        })
        
        return {
            'agreement_rate': agreement_rate,
            'total_comparisons': len(ml_predictions),
            'agreements': agreements,
            'ml_distribution': pd.Series(ml_predictions).value_counts().to_dict(),
            'rules_distribution': pd.Series(rule_predictions).value_counts().to_dict(),
            'comparison_matrix': comparison_df
        }
    
    def analyze_feature_importance(self, top_n: int = 15) -> Dict[str, Any]:
        
        if not self.ml_model.is_trained:
            return {}
        
        feature_names = list(self.ml_features[0].keys())
        importance = self.ml_model.get_feature_importance(feature_names)
        
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_features': sorted_features[:top_n],
            'all_importance': importance,
            'insights': self._generate_feature_insights(sorted_features[:top_n])
        }
    
    def _generate_feature_insights(self, top_features: List[Tuple[str, float]]) -> List[str]:
        
        insights = []
        
        for feature, importance in top_features:
            if importance > 0.05:  
                if 'ratio' in feature:
                    insights.append(f"📊 {feature} (важность: {importance:.3f}) - ключевое соотношение для выбора продукта")
                elif 'age' in feature:
                    insights.append(f"👤 Возраст клиента критически важен для рекомендаций")
                elif 'balance' in feature:
                    insights.append(f"💰 Финансовое состояние - основной фактор выбора")
                elif 'spending' in feature:
                    insights.append(f"🛒 Паттерны трат определяют подходящие продукты")
                elif 'risk' in feature:
                    insights.append(f"⚠️ Риск-профиль влияет на рекомендации")
        
        return insights
    
    def generate_ml_report(self) -> str:
        
        if not self.ml_model.is_trained:
            return "ML модель не обучена!"
        
        
        feature_analysis = self.analyze_feature_importance()
        comparison = self.compare_ml_vs_rules()
        
        report = f"""
🤖 ML-POWERED PUSH SYSTEM ОТЧЕТ
{'='*50}

📊 КАЧЕСТВО МОДЕЛИ:
- Согласие ML vs Rules: {comparison['agreement_rate']:.1%}
- Всего сравнений: {comparison['total_comparisons']}
- Количество фичей: {len(self.ml_features[0])}

🔝 ТОП-5 ВАЖНЫХ ПРИЗНАКОВ:
"""
        
        for i, (feature, importance) in enumerate(feature_analysis['top_features'][:5], 1):
            report += f"{i}. {feature}: {importance:.3f}\n"
        
        report += f"""
💡 ИНСАЙТЫ:
"""
        for insight in feature_analysis['insights'][:5]:
            report += f"• {insight}\n"
        
        report += f"""
📈 РАСПРЕДЕЛЕНИЕ ML РЕКОМЕНДАЦИЙ:
"""
        for product, count in comparison['ml_distribution'].items():
            percentage = count / comparison['total_comparisons'] * 100
            report += f"• {product}: {count} ({percentage:.1f}%)\n"
        
        return report
    
    def save_ml_models(self, filepath: str = 'models/ml_push_system.pkl'):
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.ml_model.save_model(filepath)
        print(f"✅ ML модели сохранены: {filepath}")
    
    def _create_training_labels_from_spending_patterns(self, analysis: Dict[str, Any]) -> str:
        
        
        
        total_spending = max(analysis.get('total_spending', 1), 1)
        age = analysis.get('age', 35)
        balance = analysis.get('avg_balance', 0)
        
        
        travel_ratio = (analysis.get('travel_spending', 0) + analysis.get('taxi_spending', 0)) / total_spending
        restaurant_ratio = analysis.get('restaurant_spending', 0) / total_spending  
        online_ratio = analysis.get('online_services_spending', 0) / total_spending
        fx_ratio = analysis.get('fx_operations', 0) / total_spending
        balance_to_spending = balance / total_spending
        
        
        
        
        if (travel_ratio > 0.12 and 
            (analysis.get('travel_spending', 0) + analysis.get('taxi_spending', 0)) > 150000) or \
           (travel_ratio > 0.20):  
            return 'Карта для путешествий'
            
        
        premium_score = 0
        if balance > 1500000: premium_score += 2
        if restaurant_ratio > 0.06: premium_score += 1  
        if analysis.get('status') == 'Премиальный клиент': premium_score += 1
        if analysis.get('jewelry_cosmetics_spending', 0) > 30000: premium_score += 1
        if premium_score >= 3:
            return 'Премиальная карта'
            
        
        credit_score = 0
        if online_ratio > 0.08: credit_score += 2
        if total_spending > 300000: credit_score += 1
        if analysis.get('entertainment_spending', 0) > 50000: credit_score += 1
        if age < 40: credit_score += 1  
        if len(analysis.get('spending_by_category', {})) > 5: credit_score += 1  
        if credit_score >= 3:
            return 'Кредитная карта'
            
        
        if fx_ratio > 0.03 and analysis.get('foreign_currency_spending', 0) > 25000:
            return 'Обмен валют'
            
        
        if (analysis.get('foreign_currency_spending', 0) > 40000 and balance > 800000) or \
           (fx_ratio > 0.05 and balance > 500000):
            return 'Депозит мультивалютный'
            
        
        if balance_to_spending > 8 and balance > 1800000:
            return 'Депозит сберегательный'
            
        
        if (age < 35 and balance > 400000 and 
            len(analysis.get('risk_indicators', [])) == 0 and
            total_spending < balance * 0.4):  
            return 'Инвестиции'
            
        
        if (balance > 3500000 and age > 40 and 
            balance_to_spending > 15):
            return 'Золотые слитки'
            
        
        if (balance < 150000 or 
            'negative_cash_flow' in analysis.get('risk_indicators', []) or
            analysis.get('monthly_cash_flow', 0) < 0):
            return 'Кредит наличными'
            
        
        if age < 45 and balance > 200000 and balance < 1500000:
            return 'Депозит накопительный'
            
        
        return 'Кредитная карта'
    
    def _create_behavioral_training_labels(self, analysis: Dict[str, Any], client_code: int) -> str:
        import random
        
        total_spending = max(analysis.get('total_spending', 1), 1)
        age = analysis.get('age', 35)
        balance = analysis.get('avg_balance', 0)
        
        travel_ratio = (analysis.get('travel_spending', 0) + analysis.get('taxi_spending', 0)) / total_spending
        restaurant_ratio = analysis.get('restaurant_spending', 0) / total_spending  
        online_ratio = analysis.get('online_services_spending', 0) / total_spending
        fx_ratio = analysis.get('fx_operations', 0) / total_spending
        balance_to_spending = balance / total_spending if total_spending > 0 else 0
        
        products_scores = {}
        
        products_scores['Карта для путешествий'] = (
            travel_ratio * 100 + 
            (1 if travel_ratio > 0.15 else 0) * 20 +
            (1 if analysis.get('travel_spending', 0) > 100000 else 0) * 15 +
            random.uniform(-10, 10)
        )
        
        products_scores['Премиальная карта'] = (
            (balance / 1000000) * 25 +
            restaurant_ratio * 80 +
            (1 if analysis.get('status') == 'Премиальный клиент' else 0) * 30 +
            (1 if analysis.get('jewelry_cosmetics_spending', 0) > 20000 else 0) * 15 +
            random.uniform(-15, 15)
        )
        
        products_scores['Кредитная карта'] = (
            online_ratio * 120 +
            (1 if total_spending > 250000 else 0) * 20 +
            (1 if age < 45 else 0) * 15 +
            (len(analysis.get('spending_by_category', {})) / 10) * 25 +
            random.uniform(-12, 12)
        )
        
        products_scores['Обмен валют'] = (
            fx_ratio * 150 +
            (1 if analysis.get('foreign_currency_spending', 0) > 20000 else 0) * 25 +
            random.uniform(-8, 8)
        )
        
        products_scores['Депозит сберегательный'] = (
            (balance_to_spending / 10) * 30 +
            (1 if balance > 1000000 else 0) * 35 +
            (1 if age > 35 else 0) * 15 +
            random.uniform(-10, 10)
        )
        
        products_scores['Инвестиции'] = (
            (1 if age < 40 else 0) * 25 +
            (1 if balance > 300000 else 0) * 20 +
            (1 if len(analysis.get('risk_indicators', [])) == 0 else 0) * 20 +
            (1 if total_spending < balance * 0.5 else 0) * 15 +
            random.uniform(-8, 8)
        )
        
        products_scores['Кредит наличными'] = (
            (1 if balance < 200000 else 0) * 40 +
            (1 if 'negative_cash_flow' in analysis.get('risk_indicators', []) else 0) * 30 +
            (1 if analysis.get('monthly_cash_flow', 0) < 0 else 0) * 25 +
            random.uniform(-5, 5)
        )
        
        if client_code % 7 == 0:
            products_scores[random.choice(list(products_scores.keys()))] += random.uniform(5, 25)
        
        max_product = max(products_scores.items(), key=lambda x: x[1])
        return max_product[0]
    
    def _calculate_benefit_heuristic(self, analysis: Dict[str, Any], product: str) -> float:
        
        
        if product == 'Премиальная карта':
            return analysis['total_spending'] * 0.03 
        elif product == 'Карта для путешествий':
            return (analysis['travel_spending'] + analysis['taxi_spending']) * 0.04
        elif product == 'Кредитная карта':
            return analysis['online_services_spending'] * 0.10
        elif product == 'Обмен валют':
            return analysis['fx_operations'] * 0.005 
        elif product in ['Депозит сберегательный', 'Депозит мультивалютный']:
            return analysis['avg_balance'] * 0.08 
        elif product == 'Депозит накопительный':
            return analysis['avg_balance'] * 0.06 
        elif product == 'Инвестиции':
            return analysis['avg_balance'] * 0.12 
        elif product == 'Кредит наличными':
            return 50000 
        else:
            return 10000 


def demo_ml_system():
    print("🚀 ДЕМО: ML-Powered Push System")
    print("="*60)
    
    
    ml_system = MLPushSystem()
    
    
    ml_system.load_and_prepare_data()
    
    
    training_results = ml_system.train_ml_models()
    
    
    print("\n📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"Точность классификатора: {training_results['classifier_metrics']['test_accuracy']:.3f}")
    print(f"RMSE регрессора: {training_results['regressor_metrics']['test_rmse']:.0f}")
    print(f"Количество кластеров: {len(training_results['segmentation']['analysis'])}")
    
    
    print("\n🎯 ML ПРЕДСКАЗАНИЯ:")
    test_clients = [2, 7, 13, 32, 38]
    
    for client_id in test_clients:
        try:
            result = ml_system.predict_with_ml(client_id)
            ml_pred = result['ml_prediction']
            
            print(f"\n👤 Клиент {client_id}:")
            print(f"   🤖 ML: {ml_pred['product']} (уверенность: {ml_pred['confidence']:.2f})")
            print(f"   💰 Ожидаемая выгода: {ml_pred['expected_benefit']:.0f} ₸")
            print(f"   🎯 Профиль: {ml_pred['cluster_description']}")
            print(f"   ⏰ Оптимальное время: {result['timing_optimization']['optimal_hour']}:00")
            print(f"   📱 Пуш: {ml_pred['push_notification']}")
            
        except Exception as e:
            print(f"❌ Ошибка для клиента {client_id}: {e}")
    
    
    comparison = ml_system.compare_ml_vs_rules()
    print(f"\n⚔️ ML vs RULES:")
    print(f"Согласие: {comparison['agreement_rate']:.1%}")
    
    
    feature_analysis = ml_system.analyze_feature_importance()
    print(f"\n🔝 ТОП-5 ВАЖНЫХ ПРИЗНАКОВ:")
    for i, (feature, importance) in enumerate(feature_analysis['top_features'][:5], 1):
        print(f"{i}. {feature}: {importance:.3f}")
    
    
    ml_system.save_ml_models()
    
    
    print("\n🔄 Генерация CSV с рекомендациями...")
    recommendations = []
    for client_code in ml_system.clients_data.keys():
        try:
            result = ml_system.predict_with_ml(client_code)
            ml_pred = result['ml_prediction']
            
            recommendations.append({
                'client_code': client_code,
                'product': ml_pred['product'],
                'confidence': ml_pred['confidence'],
                'expected_benefit': ml_pred['expected_benefit'],
                'cluster_description': ml_pred['cluster_description'],
                'push_notification': ml_pred['push_notification']
            })
        except Exception as e:
            print(f"❌ Ошибка CSV для клиента {client_code}: {e}")
    
    import pandas as pd
    import os
    df = pd.DataFrame(recommendations)
    output_path = 'data/processed/recommendations.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ CSV сохранен: {output_path} ({len(recommendations)} записей)")
    
    
    print("\n" + ml_system.generate_ml_report())

if __name__ == "__main__":
    demo_ml_system()

