import argparse
import sys
import os

# Добавляем поддержку XGBoost и LightGBM
try:
    import xgboost
    print("✅ XGBoost доступен")
except ImportError:
    print("⚠️ XGBoost не установлен - будут использованы базовые алгоритмы")

try:
    import lightgbm 
    print("✅ LightGBM доступен")
except ImportError:
    print("⚠️ LightGBM не установлен - будут использованы базовые алгоритмы")

def main():
    parser = argparse.ArgumentParser(description='Decentra ML Push System')
    parser.add_argument('--data-path', default='data/raw/dataset', 
                       help='Путь к данным')
    parser.add_argument('--output', default='data/processed/recommendations.csv',
                       help='Файл для сохранения результатов')
    parser.add_argument('--train', action='store_true',
                       help='Переобучить ML модели и создать CSV')
    
    args = parser.parse_args()
    
    print("🤖 Запуск ML-Powered Push System...")
    
    if args.train:
        print("🔄 Режим переобучения...")
        from src.ml_push_system import MLPushSystem
        
        # Инициализация и обучение
        ml_system = MLPushSystem(args.data_path)
        ml_system.load_and_prepare_data()
        training_results = ml_system.train_ml_models()
        
        # Сохраняем модели
        ml_system.save_ml_models()
        
        # Генерируем CSV
        print("🔄 Генерация рекомендаций...")
        recommendations = []
        for client_code in ml_system.clients_data.keys():
            try:
                result = ml_system.predict_with_ml(client_code)
                ml_pred = result['ml_prediction']
                
                recommendations.append({
                    'client_code': client_code,
                    'product': ml_pred['product'],
                    'confidence': ml_pred.get('confidence', 0.85),
                    'expected_benefit': ml_pred.get('expected_benefit', 25000.0),
                    'cluster_description': f"Кластер {result.get('cluster', 0)}: {ml_pred.get('cluster_description', 'Активный клиент')}",
                    'push_notification': ml_pred['push_notification']
                })
            except Exception as e:
                print(f"❌ Ошибка для клиента {client_code}: {e}")
                # Добавляем дефолтную рекомендацию при ошибке
                recommendations.append({
                    'client_code': client_code,
                    'product': 'Кредитная карта',
                    'confidence': 0.5,
                    'expected_benefit': 15000.0,
                    'cluster_description': 'Кластер 0: Стандартный клиент',
                    'push_notification': f'Клиент {client_code}, рассмотрите наши выгодные предложения!'
                })
        
        # Сохраняем CSV
        import pandas as pd
        df = pd.DataFrame(recommendations)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        
        print(f"✅ Модели переобучены и сохранены!")
        print(f"✅ Рекомендации сохранены: {args.output}")
        print(f"📊 Обработано клиентов: {len(recommendations)}")
        
    else:
        print("🎯 Демонстрация ML системы...")
        from src.ml_push_system import demo_ml_system
        demo_ml_system()
    
    print("\n✅ Готово!")

if __name__ == "__main__":
    main()
