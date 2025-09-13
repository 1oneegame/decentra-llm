import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import random

@dataclass
class ProductBenefit:
    product_name: str
    benefit_amount: float
    benefit_percentage: float
    signals: List[str]
    reasoning: str

class ProductAnalyzer:
    def __init__(self):
        self.products = {
            'travel_card': {
                'name': 'Карта для путешествий',
                'categories': ['Путешествия', 'Отели', 'Такси'],
                'cashback_rate': 0.04,
                'currencies': ['USD', 'EUR']
            },
            'premium_card': {
                'name': 'Премиальная карта',
                'base_cashback': 0.02,
                'premium_cashback': 0.04,
                'premium_categories': ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны'],
                'atm_fee_saving': 500,
                'transfer_fee_saving': 200
            },
            'credit_card': {
                'name': 'Кредитная карта',
                'top_categories_cashback': 0.10,
                'online_cashback': 0.10,
                'online_categories': ['Едим дома', 'Смотрим дома', 'Играем дома'],
                'grace_period_months': 2
            },
            'currency_exchange': {
                'name': 'Обмен валют',
                'spread_saving_rate': 0.005,
                'currencies': ['USD', 'EUR']
            },
            'cash_loan': {
                'name': 'Кредит наличными',
                'interest_rate': 0.15,
                'processing_fee': 10000
            },
            'multicurrency_deposit': {
                'name': 'Депозит мультивалютный',
                'interest_rate': 0.08,
                'currencies': ['USD', 'EUR', 'KZT']
            },
            'savings_deposit': {
                'name': 'Депозит сберегательный',
                'interest_rate': 0.12,
                'min_amount': 1000000
            },
            'accumulative_deposit': {
                'name': 'Депозит накопительный',
                'interest_rate': 0.10,
                'min_amount': 100000
            },
            'investment_account': {
                'name': 'Инвестиции',
                'commission_saving': 0.001,
                'min_amount': 50000
            },
            'gold_bars': {
                'name': 'Золотые слитки',
                'storage_fee': 0.02,
                'min_amount': 100000
            }
        }

    def calculate_travel_card_benefit(self, transactions: pd.DataFrame, transfers: pd.DataFrame) -> ProductBenefit:
        travel_categories = ['Путешествия', 'Отели', 'Такси']
        travel_spend = transactions[transactions['category'].isin(travel_categories)]['amount'].sum()
        
        fx_transfers = transfers[transfers['type'].isin(['fx_buy', 'fx_sell'])]
        fx_amount = fx_transfers['amount'].sum()
        
        total_benefit = travel_spend * 0.04 + fx_amount * 0.02
        
        signals = []
        if travel_spend > 50000:
            signals.append(f"Высокие траты на путешествия: {travel_spend:,.0f} KZT")
        if fx_amount > 100000:
            signals.append(f"Активный обмен валют: {fx_amount:,.0f} KZT")
        
        reasoning = f"Кешбэк 4% на путешествия ({travel_spend:,.0f} KZT) + 2% на валютные операции ({fx_amount:,.0f} KZT)"
        
        return ProductBenefit(
            product_name='Карта для путешествий',
            benefit_amount=total_benefit,
            benefit_percentage=0.04,
            signals=signals,
            reasoning=reasoning
        )

    def calculate_premium_card_benefit(self, transactions: pd.DataFrame, transfers: pd.DataFrame, 
                                     client_balance: float, client_status: str) -> ProductBenefit:
        base_spend = transactions['amount'].sum()
        premium_categories = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны']
        premium_spend = transactions[transactions['category'].isin(premium_categories)]['amount'].sum()
        
        atm_withdrawals = transfers[transfers['type'] == 'atm_withdrawal']['amount'].sum()
        card_transfers = transfers[transfers['type'] == 'card_out']['amount'].sum()
        
        if client_status == 'Премиальный клиент' and client_balance > 1000000:
            base_cashback = 0.04
        else:
            base_cashback = 0.02
            
        base_benefit = base_spend * base_cashback
        premium_benefit = premium_spend * 0.04
        fee_savings = (atm_withdrawals + card_transfers) * 0.01
        
        total_benefit = base_benefit + premium_benefit + fee_savings
        
        signals = []
        if client_balance > 1000000:
            signals.append(f"Высокий остаток: {client_balance:,.0f} KZT")
        if premium_spend > 20000:
            signals.append(f"Траты на премиум категории: {premium_spend:,.0f} KZT")
        if atm_withdrawals > 50000:
            signals.append(f"Частые снятия наличных: {atm_withdrawals:,.0f} KZT")
            
        reasoning = f"Базовый кешбэк {base_cashback*100:.0f}% ({base_spend:,.0f} KZT) + премиум кешбэк 4% ({premium_spend:,.0f} KZT) + экономия на комиссиях"
        
        return ProductBenefit(
            product_name='Премиальная карта',
            benefit_amount=total_benefit,
            benefit_percentage=base_cashback,
            signals=signals,
            reasoning=reasoning
        )

    def calculate_credit_card_benefit(self, transactions: pd.DataFrame, transfers: pd.DataFrame) -> ProductBenefit:
        category_spend = transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_categories = category_spend.head(3)
        top_categories_benefit = top_categories.sum() * 0.10
        
        online_categories = ['Едим дома', 'Смотрим дома', 'Играем дома']
        online_spend = transactions[transactions['category'].isin(online_categories)]['amount'].sum()
        online_benefit = online_spend * 0.10
        
        installment_payments = transfers[transfers['type'].isin(['installment_payment_out', 'cc_repayment_out'])]
        installment_amount = installment_payments['amount'].sum()
        
        total_benefit = top_categories_benefit + online_benefit
        
        signals = []
        if len(top_categories) > 0:
            signals.append(f"Топ категории: {', '.join(top_categories.index[:3])}")
        if online_spend > 10000:
            signals.append(f"Онлайн сервисы: {online_spend:,.0f} KZT")
        if installment_amount > 0:
            signals.append(f"Существующие рассрочки: {installment_amount:,.0f} KZT")
            
        reasoning = f"Кешбэк 10% на топ-3 категории ({top_categories.sum():,.0f} KZT) + 10% на онлайн сервисы ({online_spend:,.0f} KZT)"
        
        return ProductBenefit(
            product_name='Кредитная карта',
            benefit_amount=total_benefit,
            benefit_percentage=0.10,
            signals=signals,
            reasoning=reasoning
        )

    def calculate_deposit_benefit(self, deposit_type: str, client_balance: float, 
                                transfers: pd.DataFrame, transactions: pd.DataFrame) -> ProductBenefit:
        if deposit_type == 'multicurrency':
            fx_transfers = transfers[transfers['type'].isin(['fx_buy', 'fx_sell'])]
            fx_amount = fx_transfers['amount'].sum()
            benefit = client_balance * 0.08 + fx_amount * 0.02
            
            signals = []
            if client_balance > 500000:
                signals.append(f"Свободные средства: {client_balance:,.0f} KZT")
            if fx_amount > 50000:
                signals.append(f"Валютные операции: {fx_amount:,.0f} KZT")
                
            reasoning = f"Проценты 8% годовых + бонус за валютные операции"
            
        elif deposit_type == 'savings':
            if client_balance >= 1000000:
                benefit = client_balance * 0.12
                signals = [f"Крупный остаток: {client_balance:,.0f} KZT"]
                reasoning = "Максимальная ставка 12% годовых"
            else:
                benefit = 0
                signals = ["Недостаточный остаток для сберегательного депозита"]
                reasoning = "Требуется минимум 1,000,000 KZT"
                
        elif deposit_type == 'accumulative':
            monthly_income = transfers[transfers['direction'] == 'in']['amount'].sum() / 3
            benefit = client_balance * 0.10 + monthly_income * 0.05
            
            signals = []
            if client_balance > 100000:
                signals.append(f"Накопления: {client_balance:,.0f} KZT")
            if monthly_income > 50000:
                signals.append(f"Регулярные поступления: {monthly_income:,.0f} KZT/мес")
                
            reasoning = "Повышенная ставка 10% + бонус за пополнения"
            
        product_names = {
            'multicurrency': 'Депозит мультивалютный',
            'savings': 'Депозит сберегательный', 
            'accumulative': 'Депозит накопительный'
        }
        
        return ProductBenefit(
            product_name=product_names[deposit_type],
            benefit_amount=benefit,
            benefit_percentage=0.10 if deposit_type != 'savings' else 0.12,
            signals=signals,
            reasoning=reasoning
        )

    def calculate_investment_benefit(self, client_balance: float, transfers: pd.DataFrame) -> ProductBenefit:
        if client_balance >= 50000:
            monthly_turnover = transfers['amount'].sum() / 3
            commission_saving = monthly_turnover * 0.001
            
            signals = [f"Свободные средства: {client_balance:,.0f} KZT"]
            if monthly_turnover > 100000:
                signals.append(f"Активные операции: {monthly_turnover:,.0f} KZT/мес")
                
            reasoning = f"Экономия на комиссиях: {commission_saving:,.0f} KZT/мес"
            benefit = commission_saving * 12
        else:
            benefit = 0
            signals = ["Недостаточно средств для инвестиций"]
            reasoning = "Требуется минимум 50,000 KZT"
            
        return ProductBenefit(
            product_name='Инвестиции',
            benefit_amount=benefit,
            benefit_percentage=0.001,
            signals=signals,
            reasoning=reasoning
        )

class PushNotificationGenerator:
    def __init__(self):
        self.month_names = {
            1: 'январе', 2: 'феврале', 3: 'марте', 4: 'апреле',
            5: 'мае', 6: 'июне', 7: 'июле', 8: 'августе',
            9: 'сентябре', 10: 'октябре', 11: 'ноябре', 12: 'декабре'
        }
        
    def analyze_client_behavior(self, transactions: pd.DataFrame, transfers: pd.DataFrame, 
                              client_data: Dict) -> Dict:
        behavior = {}
        
        transactions['date'] = pd.to_datetime(transactions['date'])
        transfers['date'] = pd.to_datetime(transfers['date'])
        
        current_month = transactions['date'].dt.month.mode().iloc[0] if len(transactions) > 0 else 8
        
        behavior['current_month'] = self.month_names.get(current_month, 'августе')
        
        travel_categories = ['Путешествия', 'Отели', 'Такси']
        travel_spend = transactions[transactions['category'].isin(travel_categories)]['amount'].sum()
        behavior['travel_spend'] = travel_spend
        behavior['taxi_count'] = len(transactions[transactions['category'] == 'Такси'])
        behavior['taxi_amount'] = transactions[transactions['category'] == 'Такси']['amount'].sum()
        
        behavior['top_categories'] = transactions.groupby('category')['amount'].sum().sort_values(ascending=False).head(3)
        
        fx_transfers = transfers[transfers['type'].isin(['fx_buy', 'fx_sell'])]
        behavior['fx_amount'] = fx_transfers['amount'].sum()
        behavior['fx_currency'] = 'USD' if fx_transfers['amount'].sum() > 0 else None
        
        behavior['atm_withdrawals'] = transfers[transfers['type'] == 'atm_withdrawal']['amount'].sum()
        behavior['card_transfers'] = transfers[transfers['type'] == 'card_out']['amount'].sum()
        
        premium_categories = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны']
        behavior['premium_spend'] = transactions[transactions['category'].isin(premium_categories)]['amount'].sum()
        
        online_categories = ['Едим дома', 'Смотрим дома', 'Играем дома']
        behavior['online_spend'] = transactions[transactions['category'].isin(online_categories)]['amount'].sum()
        
        behavior['monthly_income'] = transfers[transfers['direction'] == 'in']['amount'].sum() / 3
        
        return behavior
    
    def generate_travel_card_push(self, client_data: Dict, behavior: Dict, benefit: ProductBenefit) -> str:
        name = client_data['name']
        month = behavior['current_month']
        
        if behavior['taxi_count'] > 0:
            taxi_amount = int(behavior['taxi_amount'])
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, в {month} вы сделали {behavior['taxi_count']} поездок на такси на {taxi_amount:,} ₸. С картой для путешествий вернули бы ≈{cashback_amount:,} ₸. Откройте карту в приложении."
        else:
            travel_spend = int(behavior['travel_spend'])
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, в {month} вы потратили {travel_spend:,} ₸ на путешествия. С картой для путешествий вернули бы ≈{cashback_amount:,} ₸. Откройте карту в приложении."
    
    def generate_premium_card_push(self, client_data: Dict, behavior: Dict, benefit: ProductBenefit) -> str:
        name = client_data['name']
        balance = int(client_data['avg_monthly_balance_KZT'])
        
        if behavior['premium_spend'] > 0:
            premium_spend = int(behavior['premium_spend'])
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, у вас высокий остаток {balance:,} ₸ и траты в ресторанах {premium_spend:,} ₸. Премиальная карта даст до 4% кешбэка и бесплатные снятия. Подключите сейчас."
        else:
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, у вас стабильно крупный остаток {balance:,} ₸. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."
    
    def generate_credit_card_push(self, client_data: Dict, behavior: Dict, benefit: ProductBenefit) -> str:
        name = client_data['name']
        top_cats = behavior['top_categories']
        
        if len(top_cats) >= 3:
            cat1, cat2, cat3 = top_cats.index[:3]
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту."
        else:
            online_spend = int(behavior['online_spend'])
            cashback_amount = int(benefit.benefit_amount)
            return f"{name}, вы тратите {online_spend:,} ₸ на онлайн-сервисы. Кредитная карта даёт 10% кешбэка на онлайн-покупки. Оформить карту."
    
    def generate_deposit_push(self, client_data: Dict, behavior: Dict, benefit: ProductBenefit, deposit_type: str) -> str:
        name = client_data['name']
        balance = int(client_data['avg_monthly_balance_KZT'])
        
        if deposit_type == 'multicurrency':
            fx_amount = int(behavior['fx_amount'])
            if fx_amount > 0:
                return f"{name}, вы часто обмениваете валюту на {fx_amount:,} ₸. Мультивалютный депозит даст проценты и удобство хранения валют. Открыть депозит."
            else:
                return f"{name}, у вас свободные средства {balance:,} ₸. Мультивалютный депозит даст проценты и удобство хранения валют. Открыть депозит."
        elif deposit_type == 'savings':
            return f"{name}, у вас крупный остаток {balance:,} ₸. Сберегательный депозит даст максимальную ставку 12% годовых. Открыть депозит."
        else:  # accumulative
            monthly_income = int(behavior['monthly_income'])
            return f"{name}, у вас регулярные поступления {monthly_income:,} ₸ в месяц. Накопительный депозит даст повышенную ставку и удобство пополнения. Открыть депозит."
    
    def generate_investment_push(self, client_data: Dict, behavior: Dict, benefit: ProductBenefit) -> str:
        name = client_data['name']
        balance = int(client_data['avg_monthly_balance_KZT'])
        
        return f"{name}, у вас свободные средства {balance:,} ₸. Попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт."
    
    def generate_push_notification(self, client_data: Dict, transactions: pd.DataFrame, 
                                 transfers: pd.DataFrame, product_benefit: ProductBenefit) -> str:
        behavior = self.analyze_client_behavior(transactions, transfers, client_data)
        
        product_name = product_benefit.product_name.lower()
        
        if 'путешествий' in product_name:
            return self.generate_travel_card_push(client_data, behavior, product_benefit)
        elif 'премиальная' in product_name:
            return self.generate_premium_card_push(client_data, behavior, product_benefit)
        elif 'кредитная' in product_name:
            return self.generate_credit_card_push(client_data, behavior, product_benefit)
        elif 'мультивалютный' in product_name:
            return self.generate_deposit_push(client_data, behavior, product_benefit, 'multicurrency')
        elif 'сберегательный' in product_name:
            return self.generate_deposit_push(client_data, behavior, product_benefit, 'savings')
        elif 'накопительный' in product_name:
            return self.generate_deposit_push(client_data, behavior, product_benefit, 'accumulative')
        elif 'инвестиции' in product_name:
            return self.generate_investment_push(client_data, behavior, product_benefit)
        else:
            return f"{client_data['name']}, для вас есть выгодное предложение — {product_benefit.product_name}. Оформить сейчас."

class ClientAnalyzer:
    def __init__(self):
        self.product_analyzer = ProductAnalyzer()
        self.push_generator = PushNotificationGenerator()
        
    def analyze_client(self, client_data: Dict, transactions: pd.DataFrame, transfers: pd.DataFrame) -> Dict:
        client_code = client_data['client_code']
        client_balance = client_data['avg_monthly_balance_KZT']
        client_status = client_data['status']
        
        benefits = {}
        
        benefits['travel_card'] = self.product_analyzer.calculate_travel_card_benefit(transactions, transfers)
        benefits['premium_card'] = self.product_analyzer.calculate_premium_card_benefit(
            transactions, transfers, client_balance, client_status
        )
        benefits['credit_card'] = self.product_analyzer.calculate_credit_card_benefit(transactions, transfers)
        benefits['multicurrency_deposit'] = self.product_analyzer.calculate_deposit_benefit(
            'multicurrency', client_balance, transfers, transactions
        )
        benefits['savings_deposit'] = self.product_analyzer.calculate_deposit_benefit(
            'savings', client_balance, transfers, transactions
        )
        benefits['accumulative_deposit'] = self.product_analyzer.calculate_deposit_benefit(
            'accumulative', client_balance, transfers, transactions
        )
        benefits['investment'] = self.product_analyzer.calculate_investment_benefit(client_balance, transfers)
        
        best_product = max(benefits.items(), key=lambda x: x[1].benefit_amount)
        
        push_notification = self.push_generator.generate_push_notification(
            client_data, transactions, transfers, best_product[1]
        )
        
        return {
            'client_code': client_code,
            'client_name': client_data['name'],
            'client_status': client_status,
            'client_balance': client_balance,
            'best_product': best_product[0],
            'best_product_name': best_product[1].product_name,
            'best_benefit': best_product[1].benefit_amount,
            'all_benefits': benefits,
            'recommendation_reasoning': best_product[1].reasoning,
            'signals': best_product[1].signals,
            'push_notification': push_notification
        }

class PersonalizedRecommendationModel:
    def __init__(self):
        self.client_analyzer = ClientAnalyzer()
        
    def process_client_data(self, client_file_path: str) -> Dict:
        try:
            clients_df = pd.read_csv('dataset/clients.csv')
            client_code = int(client_file_path.split('_')[1])
            client_data = clients_df[clients_df['client_code'] == client_code].iloc[0].to_dict()
            
            transactions_file = f'dataset/client_{client_code}_transactions_3m.csv'
            transfers_file = f'dataset/client_{client_code}_transfers_3m.csv'
            
            transactions = pd.read_csv(transactions_file)
            transfers = pd.read_csv(transfers_file)
            
            transactions['date'] = pd.to_datetime(transactions['date'])
            transfers['date'] = pd.to_datetime(transfers['date'])
            
            result = self.client_analyzer.analyze_client(client_data, transactions, transfers)
            
            return result
            
        except Exception as e:
            return {'error': f'Ошибка обработки данных клиента: {str(e)}'}
    
    def get_recommendation(self, client_file_path: str) -> Tuple[str, str, float]:
        result = self.process_client_data(client_file_path)
        
        if 'error' in result:
            return None, None, 0
            
        return (
            result['client_code'],
            result['best_product_name'],
            result['best_benefit']
        )
    
    def process_all_clients(self) -> pd.DataFrame:
        results = []
        
        for client_id in range(1, 61):
            try:
                result = self.process_client_data(f'client_{client_id}')
                if 'error' not in result:
                    results.append({
                        'client_code': result['client_code'],
                        'product': result['best_product_name'],
                        'push_notification': result['push_notification']
                    })
                else:
                    print(f"Ошибка обработки клиента {client_id}: {result['error']}")
            except Exception as e:
                print(f"Ошибка обработки клиента {client_id}: {e}")
        
        return pd.DataFrame(results)
    
    def export_to_csv(self, filename: str = 'recommendations.csv'):
        df = self.process_all_clients()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Результаты экспортированы в {filename}")
        print(f"Обработано клиентов: {len(df)}")
        return df

def main():
    model = PersonalizedRecommendationModel()
    
    print("=== Демонстрация работы модели ===")
    print("Обработка первых 5 клиентов:")
    print("=" * 60)
    
    for i in range(1, 6):
        result = model.process_client_data(f'client_{i}')
        print(f"\nКлиент: {result['client_name']} (ID: {result['client_code']})")
        print(f"Статус: {result['client_status']}")
        print(f"Остаток: {result['client_balance']:,.0f} KZT")
        print(f"Рекомендуемый продукт: {result['best_product_name']}")
        print(f"Ожидаемая выгода: {result['best_benefit']:,.0f} KZT")
        print(f"Пуш-уведомление: {result['push_notification']}")
        print("-" * 60)
    
    print("\n=== Обработка всех клиентов и экспорт в CSV ===")
    df = model.export_to_csv('recommendations.csv')
    
    print(f"\nСтатистика по продуктам:")
    product_stats = df['product'].value_counts()
    print(product_stats)
    
    print(f"\nПримеры пуш-уведомлений:")
    for i, row in df.head(3).iterrows():
        print(f"Клиент {row['client_code']}: {row['push_notification']}")

if __name__ == "__main__":
    main()
