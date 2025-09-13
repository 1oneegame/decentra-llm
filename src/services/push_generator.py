from typing import Dict, Any
import random

class PushGenerator:    
    def __init__(self):
        self.templates = {
            'Карта для путешествий': [
                "{name}, в {month} у вас много поездок/такси — {travel_amount:,.0f} ₸. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?",
                "{name}, заметили активные поездки на {travel_amount:,.0f} ₸. Карта для путешествий даст 4% кешбэка. Открыть карту."
            ],
            'Премиальная карта': [
                "{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас.",
                "{name}, остаток {balance:,.0f} ₸ открывает премиальные возможности. До 4% кешбэка и бесплатные переводы. Подключить карту."
            ],
            'Кредитная карта': [
                "{name}, ваши топ-категории — {categories}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту.",
                "{name}, активные траты в {top_category} и онлайн-сервисах. Кредитка вернет до 10% кешбэка. Открыть карту."
            ],
            'Обмен валют': [
                "{name}, вы часто платите в {fx_curr}. В приложении выгодный обмен и авто-покупка по целевому курсу. Настроить обмен.",
                "{name}, валютные операции на {fx_amount:,.0f} ₸. Выгодный курс 24/7 без комиссии в приложении. Настроить обмен."
            ],
            'Депозит мультивалютный': [
                "{name}, у вас остаются свободные средства в валюте. Мультивалютный депозит 14,5% — удобно копить. Открыть вклад.",
                "{name}, валютные операции + высокий баланс. Депозит в валюте с пополнением и снятием. Оформить депозит."
            ],
            'Депозит сберегательный': [
                "{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.",
                "{name}, высокий остаток {balance:,.0f} ₸ может приносить 16,5% годовых. Сберегательный депозит. Открыть вклад."
            ],
            'Депозит накопительный': [
                "{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад.",
                "{name}, накопительный депозит 15,5% — можно пополнять, нельзя снимать. Планомерно копить. Открыть вклад."
            ],
            'Инвестиции': [
                "{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт.",
                "{name}, молодой возраст + хороший баланс = время для инвестиций. Комиссии 0% в первый год. Открыть счёт."
            ],
            'Золотые слитки': [
                "{name}, высокий капитал требует диверсификации. Золотые слитки 999,9 пробы для сохранения стоимости. Узнать подробнее.",
                "{name}, золото — защитный актив в портфеле. Слитки разных весов, хранение в банке. Оформить покупку."
            ],
            'Кредит наличными': [
                "{name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать доступный лимит.",
                "{name}, кредит без справок и залога. 12% на год, досрочное погашение без штрафов. Проверить лимит."
            ]
        }
    
    def generate_push(self, client_analysis: Dict[str, Any], product: str) -> str:        
        if product not in self.templates:
            return f"{client_analysis['name']}, персональная рекомендация: {product}. Узнать подробнее."
        
        template = random.choice(self.templates[product])
        
        params = {
            'name': client_analysis.get('name', 'Клиент'),
            'balance': client_analysis.get('avg_balance', 0),
            'month': 'этом месяце',
            'travel_amount': client_analysis.get('travel_spending', 0) + client_analysis.get('taxi_spending', 0),
            'cashback': (client_analysis.get('travel_spending', 0) + client_analysis.get('taxi_spending', 0)) * 0.04,
            'fx_amount': client_analysis.get('fx_operations', 0),
            'fx_curr': 'USD/EUR',
            'top_category': self._get_top_category(client_analysis),
            'categories': self._get_top_categories_string(client_analysis)
        }
        
        try:
            return template.format(**params)
        except KeyError as e:
            return f"{params['name']}, персональная рекомендация: {product}. Узнать подробнее."
    
    def _get_top_category(self, analysis: Dict[str, Any]) -> str:
        spending_by_category = {}
        
        category_translation = {
            'total': 'Общие траты',
            'restaurant': 'Рестораны',
            'travel': 'Путешествия', 
            'taxi': 'Такси',
            'online_services': 'Онлайн-сервисы',
            'foreign_currency': 'Валютные операции',
            'entertainment': 'Развлечения',
            'food': 'Продукты питания',
            'medical': 'Медицина',
            'auto': 'Авто',
            'jewelry_cosmetics': 'Ювелирка и косметика',
            'atm_withdrawals': 'Снятие наличных'
        }
        
        for key, value in analysis.items():
            if key.endswith('_spending') and value > 0:
                category_key = key.replace('_spending', '')
                category_name = category_translation.get(category_key, category_key.replace('_', ' ').title())
                spending_by_category[category_name] = value
        
        if spending_by_category:
            return max(spending_by_category.items(), key=lambda x: x[1])[0]
        
        return "Продукты питания"
    
    def _get_top_categories_string(self, analysis: Dict[str, Any]) -> str:
        spending_by_category = {}
        
        category_translation = {
            'total': 'Общие траты',
            'restaurant': 'Рестораны',
            'travel': 'Путешествия', 
            'taxi': 'Такси',
            'online_services': 'Онлайн-сервисы',
            'foreign_currency': 'Валютные операции',
            'entertainment': 'Развлечения',
            'food': 'Продукты питания',
            'medical': 'Медицина',
            'auto': 'Авто',
            'jewelry_cosmetics': 'Ювелирка и косметика',
            'atm_withdrawals': 'Снятие наличных'
        }
        
        for key, value in analysis.items():
            if key.endswith('_spending') and value > 0:
                category_key = key.replace('_spending', '')
                category_name = category_translation.get(category_key, category_key.replace('_', ' ').title())
                spending_by_category[category_name] = value
        
        if spending_by_category:
            top_3 = sorted(spending_by_category.items(), key=lambda x: x[1], reverse=True)[:3]
            return ', '.join([cat for cat, _ in top_3])
        
        return "Продукты питания, Кафе и рестораны, Одежда и обувь"