# ML Push System API

FastAPI эндпоинт для анализа отдельного пользователя с использованием ML системы.

## Запуск

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Запустите сервер:

```bash
python run_server.py
```

3. API будет доступно по адресу: http://localhost:8000

### Оптимизация запуска

- **Первый запуск**: Модель обучается и сохраняется в `models/ml_push_system.pkl`
- **Последующие запуски**: Модель загружается из сохраненного файла (быстрее)
- **Принудительное переобучение**: Установите переменную окружения `FORCE_RETRAIN=true`

## Эндпоинты

### GET /

Проверка статуса API

```json
{
  "status": "ok",
  "message": "ML Push System API работает"
}
```

### GET /health

Проверка готовности ML системы

```json
{
  "status": "healthy",
  "ml_system_ready": true
}
```

### POST /retrain

Принудительное переобучение модели

**Ответ:**

```json
{
  "status": "success",
  "message": "Модель успешно переобучена",
  "training_results": {
    "classifier_accuracy": 0.85,
    "regressor_rmse": 12000.0,
    "clusters_count": 6
  }
}
```

### POST /predict

Анализ отдельного пользователя (из существующих данных)

**Запрос:**

```json
{
  "client_code": 2
}
```

### POST /analyze-client

Анализ клиента по загруженным файлам

**Запрос:**

- `client_data` (form field): JSON с данными клиента
- `transactions_file` (file): CSV файл с транзакциями
- `transfers_file` (file): CSV файл с переводами

**Пример client_data:**

```json
{
  "client_code": 999,
  "name": "Иван Иванов",
  "status": "Обычный клиент",
  "age": 28,
  "city": "Алматы",
  "avg_monthly_balance_KZT": 450000.0
}
```

**Формат CSV файлов:**

- transactions: `date,category,amount,currency`
- transfers: `date,type,direction,amount,currency`

### POST /predict-push

Упрощенный эндпоинт для предсказания пуш-уведомления

**Запрос:**

- `client_data` (form field): JSON с данными клиента
- `transactions_file` (file): CSV файл с транзакциями
- `transfers_file` (file): CSV файл с переводами

**Ответ:**

```json
{
  "client_code": 999,
  "push_notification": "Иван Иванов, выгодно и практично! Траты Общие траты, Путешествия, Такси = возврат 20,182 ₸. Без скрытых комиссий!",
  "recommended_product": "Кредитная карта",
  "confidence": 0.85,
  "optimal_time": 14
}
```

### POST /predict-push/{client_code}

Предсказание пуш-уведомления для существующего клиента

**Параметры:**

- `client_code` (path): ID клиента из базы данных

**Описание:**
Эндпоинт автоматически загружает данные клиента и его транзакции/переводы из файлов:

- `data/raw/dataset/clients.csv` - данные клиента
- `data/raw/dataset/client_{client_code}_transactions_3m.csv` - транзакции клиента
- `data/raw/dataset/client_{client_code}_transfers_3m.csv` - переводы клиента

**Ответ:**

```json
{
  "client_code": 1,
  "push_notification": "Айгерим, оптимизируйте ликвидность! Кредитное плечо для инвестиционных возможностей. Гибкие условия возврата.",
  "recommended_product": "Кредит наличными",
  "confidence": 0.979403740266614,
  "optimal_time": 14
}
```

**Возможные ошибки:**

- `404` - Клиент не найден
- `404` - Файлы транзакций/переводов не найдены
- `500` - ML система не инициализирована

## Эндпоинты для работы с рекомендациями

### GET /recommendations

Получение всех рекомендаций из файла `recommendations.csv`

**Ответ:**

```json
[
  {
    "client_code": 1,
    "product": "Кредит наличными",
    "confidence": 0.979403740266614,
    "expected_benefit": 47387.223999278685,
    "cluster_description": "высокодоходные инвесторы",
    "push_notification": "Айгерим, оптимизируйте ликвидность! Кредитное плечо для инвестиционных возможностей. Гибкие условия возврата."
  }
]
```

### GET /recommendations/{client_code}

Получение рекомендации для конкретного клиента

**Параметры:**

- `client_code` (path): ID клиента

**Ответ:**

```json
{
  "client_code": 2,
  "product": "Кредитная карта",
  "confidence": 0.8929055581895223,
  "expected_benefit": 47838.75569342719,
  "cluster_description": "премиальные клиенты",
  "push_notification": "Данияр, эксклюзивная кредитная карта для вашего статуса. 47,291 ₸ возврата + консьерж-сервис 24/7."
}
```

## Эндпоинты для работы с клиентами

### GET /clients

Получение всех клиентов из файла `clients.csv`

**Ответ:**

```json
[
  {
    "client_code": 1,
    "name": "Айгерим",
    "status": "Зарплатный клиент",
    "age": 29,
    "city": "Алматы",
    "avg_monthly_balance_KZT": 92643.0
  },
  {
    "client_code": 2,
    "name": "Данияр",
    "status": "Премиальный клиент",
    "age": 41,
    "city": "Астана",
    "avg_monthly_balance_KZT": 1577073.0
  }
]
```

### GET /clients/{client_code}

Получение клиента по ID

**Параметры:**

- `client_code` (path): ID клиента

**Ответ:**

```json
{
  "client_code": 2,
  "name": "Данияр",
  "status": "Премиальный клиент",
  "age": 41,
  "city": "Астана",
  "avg_monthly_balance_KZT": 1577073.0
}
```

### GET /clients/stats

Получение статистики по клиентам

**Ответ:**

```json
{
  "total_clients": 60,
  "status_distribution": {
    "Зарплатный клиент": 25,
    "Премиальный клиент": 20,
    "Стандартный клиент": 10,
    "Студент": 5
  },
  "city_distribution": {
    "Алматы": 30,
    "Астана": 15,
    "Шымкент": 8,
    "Караганда": 4,
    "Павлодар": 3
  },
  "age_stats": {
    "min": 20,
    "max": 58,
    "mean": 35.5,
    "median": 34.0
  },
  "balance_stats": {
    "min": 46630.0,
    "max": 5818675.0,
    "mean": 850000.0,
    "median": 120000.0
  }
}
```

## Тестирование

### Тестирование существующих клиентов:

```bash
python test_api.py
```

### Тестирование загрузки файлов:

```bash
python test_file_upload.py
```

### Тестирование предсказания пуш-уведомлений:

```bash
python test_push_prediction.py
```

### Примеры файлов:

- `example_transactions.csv` - пример файла транзакций
- `example_transfers.csv` - пример файла переводов

## Структура ответа

- `client_code` - ID клиента
- `ml_prediction.product` - рекомендованный продукт
- `ml_prediction.confidence` - уверенность модели (0-1)
- `ml_prediction.expected_benefit` - ожидаемая выгода в тенге
- `ml_prediction.cluster` - номер кластера клиента
- `ml_prediction.cluster_description` - описание кластера
- `ml_prediction.push_notification` - текст пуш-уведомления
- `timing_optimization.optimal_hour` - оптимальный час для отправки
- `features_used` - использованные признаки для предсказания

## Обработка ошибок

- `404` - Клиент не найден
- `500` - Ошибка ML системы или сервера
