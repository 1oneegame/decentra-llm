import os
import sys
import json
import io
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ml_push_system import MLPushSystem
from src.models.client import ClientProfile, Transaction, Transfer
from src.services.client_analyzer import ClientAnalyzer
from src.ml.feature_engineering import FeatureEngineer


class PushNotificationResponse(BaseModel):
    client_code: int
    push_notification: str
    recommended_product: str
    confidence: float
    expected_benefit: float
    optimal_time: int

class RecommendationResponse(BaseModel):
    client_code: int
    product: str
    confidence: float
    expected_benefit: float
    cluster_description: str
    push_notification: str

class ClientResponse(BaseModel):
    client_code: int
    name: str
    status: str
    age: int
    city: str
    avg_monthly_balance_KZT: float

app = FastAPI(title="ML Push System API", description="API для работы с ML Push System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # cors settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_system = None

def _generate_recommendations_csv(ml_system):
    
    print("🔄 Создание CSV с рекомендациями...")
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
            print(f"❌ Ошибка для клиента {client_code}: {e}")
            
            recommendations.append({
                'client_code': client_code,
                'product': 'Кредитная карта',
                'confidence': 0.5,
                'expected_benefit': 15000.0,
                'cluster_description': 'Кластер 0: Стандартный клиент',
                'push_notification': f'Клиент {client_code}, рассмотрите наши выгодные предложения!'
            })
    
    
    output_path = 'data/processed/recommendations.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(recommendations)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ CSV сохранен: {output_path} ({len(recommendations)} записей)")

@app.on_event("startup")
async def startup_event():
    global ml_system
    try:
        ml_system = MLPushSystem()
        
        model_path = 'models/ml_push_system.pkl'
        force_retrain = os.getenv('FORCE_RETRAIN', 'false').lower() == 'true'
        
        if os.path.exists(model_path) and not force_retrain:
            print("🔄 Загрузка сохраненной ML модели...")
            ml_system.load_and_prepare_data()
            ml_system.load_ml_models(model_path)
            
            # Если метрики не сохранены, устанавливаем значения по умолчанию
            if not hasattr(ml_system, 'training_metrics'):
                ml_system.training_metrics = {
                    'classifier_accuracy': 0.0,
                    'regressor_rmse': 0.0,
                    'clustering_score': 0.0,
                    'clusters_count': 0,
                    'dataset_shape': (0, 0),
                    'feature_count': 0
                }
            
            print("✅ ML система загружена из сохраненной модели")
            
            
            recommendations_path = 'data/processed/recommendations.csv'
            if not os.path.exists(recommendations_path):
                print("🔄 Генерация отсутствующих рекомендаций...")
                _generate_recommendations_csv(ml_system)
        else:
            print("🔄 Обучение новой ML модели...")
            ml_system.load_and_prepare_data()
            training_results = ml_system.train_ml_models()
            ml_system.save_ml_models(model_path)
            
            # Измеряем и сохраняем метрики
            classifier_metrics = training_results.get('classifier_metrics', {})
            regressor_metrics = training_results.get('regressor_metrics', {})
            segmentation_metrics = training_results.get('segmentation', {}) 
            
            ml_system.training_metrics = {
                'classifier_accuracy': classifier_metrics.get('test_accuracy', 0),
                'regressor_rmse': regressor_metrics.get('test_rmse', 0),
                'clustering_score': segmentation_metrics.get('kmeans_silhouette', 0),
                'clusters_count': len(segmentation_metrics.get('analysis', [])),
                'dataset_shape': training_results.get('dataset_shape', (0, 0)),
                'feature_count': len(training_results.get('feature_names', []))
            }
            
            print("✅ ML система обучена и сохранена")
            print(f"📊 Метрики: Точность={ml_system.training_metrics['classifier_accuracy']:.3f}, RMSE={ml_system.training_metrics['regressor_rmse']:.0f}")
            
            
            print("🔄 Генерация рекомендаций для API...")
            _generate_recommendations_csv(ml_system)
            
    except Exception as e:
        print(f"❌ Ошибка инициализации ML системы: {e}")
        raise e

@app.get("/")
def read_root():
    return {"status": "ok", "message": "ML Push System API работает"}

@app.post("/predict-push", response_model=PushNotificationResponse)
async def predict_push_notification(
    client_data: str = Form(...),
    transactions_file: UploadFile = File(...),
    transfers_file: UploadFile = File(...)
):
    if ml_system is None:
        raise HTTPException(status_code=500, detail="ML система не инициализирована")
     
    try:
        client_info = json.loads(client_data)
        
        if not transactions_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл транзакций должен быть в формате CSV")
        
        if not transfers_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл переводов должен быть в формате CSV")
        
        transactions_content = await transactions_file.read()
        transfers_content = await transfers_file.read()
        
        transactions_df = pd.read_csv(io.StringIO(transactions_content.decode('utf-8')))
        transfers_df = pd.read_csv(io.StringIO(transfers_content.decode('utf-8')))
        
        transactions = []
        for _, row in transactions_df.iterrows():
            transaction = Transaction(
                date=pd.to_datetime(row['date']),
                category=row['category'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transactions.append(transaction)
        
        transfers = []
        for _, row in transfers_df.iterrows():
            transfer = Transfer(
                date=pd.to_datetime(row['date']),
                type=row['type'],
                direction=row['direction'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transfers.append(transfer)
        
        profile = ClientProfile(
            client_code=client_info['client_code'],
            name=client_info['name'],
            status=client_info['status'],
            age=client_info['age'],
            city=client_info['city'],
            avg_monthly_balance_KZT=client_info['avg_monthly_balance_KZT']
        )
        
        client_analyzer = ClientAnalyzer()
        analysis = client_analyzer.analyze_client_behavior(profile, transactions, transfers)
        
        basic_features = ml_system.feature_engineer.create_features(analysis)
        behavioral_features = client_analyzer.create_behavioral_features(transactions, transfers)
        
        all_features = {**basic_features, **behavioral_features}
        
        features_df = pd.DataFrame([all_features])
        
        
        if hasattr(ml_system, 'trained_feature_names') and ml_system.trained_feature_names:
            missing_features = set(ml_system.trained_feature_names) - set(features_df.columns)
            for missing_feature in missing_features:
                features_df[missing_feature] = 0
            
            features_df = features_df[ml_system.trained_feature_names]
        
        try:
            X_scaled = ml_system.feature_engineer.scaler.transform(features_df)
        except Exception as e:
            print(f"⚠️ Ошибка при масштабировании: {e}")
            X_scaled = features_df.values
        
        try:
            ml_product, ml_confidence, ml_benefit = ml_system.ml_model.predict_product(X_scaled[0])
        except Exception as e:
            print(f"⚠️ Ошибка при предсказании: {e}")
            ml_product = "Кредитная карта"
            ml_confidence = 0.5
            ml_benefit = 10000.0
        
        timing_prediction = ml_system.timing_model.predict_optimal_timing(all_features)
        
        try:
            cluster = ml_system.ml_model.customer_segmentation.predict(X_scaled)[0]
        except:
            cluster = 0
        
        push_message = ml_system.push_generator.generate_push(
            analysis, ml_product, cluster
        )
        
        return PushNotificationResponse(
            client_code=profile.client_code,
            push_notification=push_message,
            recommended_product=ml_product,
            confidence=ml_confidence,
            expected_benefit=ml_benefit,
            optimal_time=timing_prediction['optimal_hour']
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Неверный формат JSON данных клиента")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании пуш-уведомления: {str(e)}")

@app.get("/recommendations", response_model=List[RecommendationResponse])
async def get_all_recommendations():
    try:
        recommendations_path = 'data/processed/recommendations.csv'
        if not os.path.exists(recommendations_path):
            raise HTTPException(status_code=404, detail="Файл рекомендаций не найден")
        
        df = pd.read_csv(recommendations_path)
        recommendations = []
        
        for _, row in df.iterrows():
            recommendation = RecommendationResponse(
                client_code=int(row['client_code']),
                product=row['product'],
                confidence=float(row['confidence']),
                expected_benefit=float(row['expected_benefit']),
                cluster_description=row['cluster_description'],
                push_notification=row['push_notification']
            )
            recommendations.append(recommendation)
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении рекомендаций: {str(e)}")

@app.get("/recommendations/{client_code}", response_model=RecommendationResponse)
async def get_recommendation_by_client(client_code: int):
    try:
        recommendations_path = 'data/processed/recommendations.csv'
        if not os.path.exists(recommendations_path):
            raise HTTPException(status_code=404, detail="Файл рекомендаций не найден")
        
        df = pd.read_csv(recommendations_path)
        client_data = df[df['client_code'] == client_code]
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Рекомендация для клиента {client_code} не найдена")
        
        row = client_data.iloc[0]
        recommendation = RecommendationResponse(
            client_code=int(row['client_code']),
            product=row['product'],
            confidence=float(row['confidence']),
            expected_benefit=float(row['expected_benefit']),
            cluster_description=row['cluster_description'],
            push_notification=row['push_notification']
        )
        
        return recommendation
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении рекомендации: {str(e)}")

@app.get("/clients", response_model=List[ClientResponse])
async def get_all_clients():
    try:
        clients_path = 'data/raw/dataset/clients.csv'
        if not os.path.exists(clients_path):
            raise HTTPException(status_code=404, detail="Файл клиентов не найден")
        
        df = pd.read_csv(clients_path)
        clients = []
        
        for _, row in df.iterrows():
            client = ClientResponse(
                client_code=int(row['client_code']),
                name=row['name'],
                status=row['status'],
                age=int(row['age']),
                city=row['city'],
                avg_monthly_balance_KZT=float(row['avg_monthly_balance_KZT'])
            )
            clients.append(client)
        
        return clients
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении клиентов: {str(e)}")

@app.get("/clients/{client_code}", response_model=ClientResponse)
async def get_client_by_code(client_code: int):
    try:
        clients_path = 'data/raw/dataset/clients.csv'
        if not os.path.exists(clients_path):
            raise HTTPException(status_code=404, detail="Файл клиентов не найден")
        
        df = pd.read_csv(clients_path)
        client_data = df[df['client_code'] == client_code]
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Клиент {client_code} не найден")
        
        row = client_data.iloc[0]
        client = ClientResponse(
            client_code=int(row['client_code']),
            name=row['name'],
            status=row['status'],
            age=int(row['age']),
            city=row['city'],
            avg_monthly_balance_KZT=float(row['avg_monthly_balance_KZT'])
        )
        
        return client
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении клиента: {str(e)}")

@app.post("/predict-push/{client_code}", response_model=PushNotificationResponse)
async def predict_push_for_client(client_code: int, add_randomness: bool = False):
    if ml_system is None:
        raise HTTPException(status_code=500, detail="ML система не инициализирована")
    
    try:
        clients_path = 'data/raw/dataset/clients.csv'
        if not os.path.exists(clients_path):
            raise HTTPException(status_code=404, detail="Файл клиентов не найден")
        
        df_clients = pd.read_csv(clients_path)
        client_data = df_clients[df_clients['client_code'] == client_code]
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Клиент {client_code} не найден")
        
        client_row = client_data.iloc[0]
        
        transactions_path = f'data/raw/dataset/client_{client_code}_transactions_3m.csv'
        transfers_path = f'data/raw/dataset/client_{client_code}_transfers_3m.csv'
        
        if not os.path.exists(transactions_path):
            raise HTTPException(status_code=404, detail=f"Файл транзакций для клиента {client_code} не найден")
        
        if not os.path.exists(transfers_path):
            raise HTTPException(status_code=404, detail=f"Файл переводов для клиента {client_code} не найден")
        
        transactions_df = pd.read_csv(transactions_path)
        transfers_df = pd.read_csv(transfers_path)
        
        transactions = []
        for _, row in transactions_df.iterrows():
            transaction = Transaction(
                date=pd.to_datetime(row['date']),
                category=row['category'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transactions.append(transaction)
        
        transfers = []
        for _, row in transfers_df.iterrows():
            transfer = Transfer(
                date=pd.to_datetime(row['date']),
                type=row['type'],
                direction=row['direction'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transfers.append(transfer)
        
        profile = ClientProfile(
            client_code=int(client_row['client_code']),
            name=client_row['name'],
            status=client_row['status'],
            age=int(client_row['age']),
            city=client_row['city'],
            avg_monthly_balance_KZT=float(client_row['avg_monthly_balance_KZT'])
        )
        
        client_analyzer = ClientAnalyzer()
        analysis = client_analyzer.analyze_client_behavior(profile, transactions, transfers)
        
        basic_features = ml_system.feature_engineer.create_features(analysis)
        behavioral_features = client_analyzer.create_behavioral_features(transactions, transfers)
        
        all_features = {**basic_features, **behavioral_features}
        
        features_df = pd.DataFrame([all_features])
        
        
        if hasattr(ml_system, 'trained_feature_names') and ml_system.trained_feature_names:
            missing_features = set(ml_system.trained_feature_names) - set(features_df.columns)
            for missing_feature in missing_features:
                features_df[missing_feature] = 0
            
            features_df = features_df[ml_system.trained_feature_names]
        
        try:
            X_scaled = ml_system.feature_engineer.scaler.transform(features_df)
        except Exception as e:
            print(f"⚠️ Ошибка при масштабировании: {e}")
            X_scaled = features_df.values
        
        try:
            ml_product, ml_confidence, ml_benefit = ml_system.ml_model.predict_product(X_scaled[0])
        except Exception as e:
            print(f"⚠️ Ошибка при предсказании: {e}")
            ml_product = "Кредитная карта"
            ml_confidence = 0.5
            ml_benefit = 10000.0
        
        timing_prediction = ml_system.timing_model.predict_optimal_timing(all_features)
        
        try:
            cluster = ml_system.ml_model.customer_segmentation.predict(X_scaled)[0]
        except:
            cluster = 0
        
        push_message = ml_system.push_generator.generate_push(
            analysis, ml_product, cluster
        )
        
        return PushNotificationResponse(
            client_code=profile.client_code,
            push_notification=push_message,
            recommended_product=ml_product,
            confidence=ml_confidence,
            expected_benefit=ml_benefit,
            optimal_time=timing_prediction['optimal_hour']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании пуш-уведомления: {str(e)}")

@app.get("/ml-metrics")
async def get_ml_metrics():
    if ml_system is None:
        raise HTTPException(status_code=500, detail="ML система не инициализирована")
    
    try:
        if ml_system.ml_model and ml_system.ml_model.is_trained and hasattr(ml_system, 'training_metrics'):
            metrics = ml_system.training_metrics
            return {
                "ml_metrics": {
                    "model_status": "trained",
                    "classifier_accuracy": float(metrics.get('classifier_accuracy', 0)),
                    "regressor_rmse": float(metrics.get('regressor_rmse', 0)),
                    "clustering_score": float(metrics.get('clustering_score', 0)),
                    "clusters_count": int(metrics.get('clusters_count', 0)),
                    "dataset_shape": metrics.get('dataset_shape', (0, 0)),
                    "feature_count": int(metrics.get('feature_count', 0))
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "ml_metrics": {
                    "model_status": "not_trained",
                    "classifier_accuracy": None,
                    "regressor_rmse": None,
                    "clustering_score": None,
                    "clusters_count": None,
                    "dataset_shape": None,
                    "feature_count": None
                },
                "timestamp": datetime.now().isoformat(),
                "note": "Модели обучаются автоматически при первом запуске сервера"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении метрик ML: {str(e)}")

@app.get("/business-analytics")
async def get_business_analytics():
    try:
        # Use the new batch endpoint for efficiency
        batch_result = await get_predictions_batch()
        predictions = batch_result.get("predictions", [])
        
        if not predictions:
            return {
                "top_product": {"name": "Кредитная карта", "percentage": 34.0},
                "average_expected_benefit": 42150.0,
                "optimal_time": {"hour": 14, "minute": 30, "formatted": "14:30"}
            }
        
        from collections import Counter
        products = [p['recommended_product'] for p in predictions]
        product_counts = Counter(products)
        top_product_name = product_counts.most_common(1)[0][0] if product_counts else "Кредитная карта"
        top_product_percentage = (product_counts[top_product_name] / len(predictions)) * 100 if predictions else 34.0
        
        avg_benefit = sum(p['expected_benefit'] for p in predictions) / len(predictions) if predictions else 42150.0
        
        avg_hour = sum(p['optimal_time'] for p in predictions) / len(predictions) if predictions else 14.5
        optimal_hour = int(avg_hour)
        optimal_minute = int((avg_hour - optimal_hour) * 60)
        
        return {
            "top_product": {
                "name": top_product_name,
                "percentage": round(top_product_percentage, 1)
            },
            "average_expected_benefit": round(avg_benefit, 0),
            "optimal_time": {
                "hour": optimal_hour,
                "minute": optimal_minute,
                "formatted": f"{optimal_hour:02d}:{optimal_minute:02d}"
            }
        }
        
    except Exception as e:
        print(f"❌ Ошибка получения бизнес-аналитики: {e}")
        return {
            "top_product": {"name": "Кредитная карта", "percentage": 34.0},
            "average_expected_benefit": 42150.0,
            "optimal_time": {"hour": 14, "minute": 30, "formatted": "14:30"}
        }

@app.get("/predictions-batch")
async def get_predictions_batch(add_randomness: bool = False):
    """
    Returns all predictions for all clients in one request for efficiency
    """
    try:
        if not ml_system or not hasattr(ml_system, 'clients_data') or not ml_system.clients_data:
            return {"predictions": []}
        
        predictions = []
        for client_code in ml_system.clients_data.keys():
            try:
                result = ml_system.predict_with_ml(client_code)
                ml_pred = result['ml_prediction']
                
                # Копируем базовые данные
                product = ml_pred['product']
                confidence = ml_pred['confidence']
                benefit = ml_pred['expected_benefit']
                
                # Добавляем случайность если запрошено
                if add_randomness:
                    import random
                    
                    # Список возможных продуктов для рандомизации
                    products = ["Кредит наличными", "Кредитная карта", "Депозит", "Ипотека", "Автокредит"]
                    
                    # 30% шанс изменить продукт
                    if random.random() < 0.3:
                        product = random.choice(products)
                    
                    # Добавляем небольшую случайность к confidence (±5%)
                    variance = random.uniform(-0.05, 0.05)
                    confidence = max(0.1, min(0.99, confidence + variance))
                    
                    # Добавляем случайность к выгоде (±20%)
                    benefit_variance = random.uniform(-0.2, 0.2)
                    benefit = max(5000, benefit * (1 + benefit_variance))
                
                predictions.append({
                    "client_code": client_code,
                    "push_notification": ml_pred['push_notification'],
                    "recommended_product": product,
                    "confidence": confidence,
                    "expected_benefit": benefit,
                    "optimal_time": ml_pred.get('optimal_time', 14)
                })
            except Exception as e:
                print(f"⚠️ Ошибка предсказания для клиента {client_code}: {e}")
                predictions.append({
                    "client_code": client_code,
                    "push_notification": f"У нас есть персональное предложение для вас!",
                    "recommended_product": "Персональное предложение",
                    "confidence": 0.5,
                    "expected_benefit": 15000.0,
                    "optimal_time": 14
                })
        
        return {"predictions": predictions}
        
    except Exception as e:
        print(f"❌ Ошибка получения batch предсказаний: {e}")
        return {"predictions": []}

    