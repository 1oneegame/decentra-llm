from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Product:
    name: str
    benefits: List[str]
    signals: List[str]
    config: Dict[str, Any]

class ProductCatalog:
    
    @staticmethod
    def get_all_products() -> Dict[str, Product]:
        return {
            'Карта для путешествий': Product(
                name='Карта для путешествий',
                benefits=['Путешествия', 'Такси', 'Отели'],
                signals=['USD', 'EUR', 'Путешествия', 'Отели', 'Такси'],
                config={
                    'cashback_rate': 0.04,
                    'foreign_currency': True,
                    'travel_related': True
                }
            ),
            
            'Премиальная карта': Product(
                name='Премиальная карта',
                benefits=['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны'],
                signals=['high_balance', 'restaurants', 'jewelry', 'cosmetics', 'atm_usage'],
                config={
                    'base_cashback': 0.02,
                    'premium_cashback': 0.04,
                    'free_atm': True,
                    'min_balance': 1000000
                }
            ),
            
            'Кредитная карта': Product(
                name='Кредитная карта',
                benefits=['Едим дома', 'Смотрим дома', 'Играем дома'],
                signals=['varied_spending', 'online_services', 'installments'],
                config={
                    'top_categories_cashback': 0.10,
                    'online_services_cashback': 0.10,
                    'grace_period': 60,
                    'installments': True
                }
            ),
            
            'Обмен валют': Product(
                name='Обмен валют',
                benefits=[],
                signals=['fx_operations', 'foreign_spending'],
                config={
                    'spread_savings': True,
                    'auto_exchange': True,
                    'foreign_currency': True
                }
            ),
            
            'Кредит наличными': Product(
                name='Кредит наличными',
                benefits=[],
                signals=['cash_gap', 'low_balance', 'loan_payments'],
                config={
                    'quick_access': True,
                    'flexible_payments': True,
                    'risk_product': True
                }
            ),
            
            'Депозит мультивалютный': Product(
                name='Депозит мультивалютный',
                benefits=[],
                signals=['free_balance', 'fx_activity', 'foreign_spending'],
                config={
                    'interest_rate': 0.08,
                    'multi_currency': True,
                    'flexible': True
                }
            ),
            
            'Депозит сберегательный': Product(
                name='Депозит сберегательный',
                benefits=[],
                signals=['stable_balance', 'low_volatility'],
                config={
                    'interest_rate': 0.10,
                    'locked': True,
                    'high_rate': True
                }
            ),
            
            'Депозит накопительный': Product(
                name='Депозит накопительный', 
                benefits=[],
                signals=['regular_savings', 'growing_balance'],
                config={
                    'interest_rate': 0.09,
                    'top_up': True,
                    'no_withdrawal': True
                }
            ),
            
            'Инвестиции': Product(
                name='Инвестиции',
                benefits=[],
                signals=['free_money', 'risk_tolerance'],
                config={
                    'low_fees': True,
                    'low_entry': True,
                    'growth_potential': True
                }
            ),
            
            'Золотые слитки': Product(
                name='Золотые слитки',
                benefits=[],
                signals=['high_liquidity', 'jewelry_interest', 'wealth_protection'],
                config={
                    'hedge': True,
                    'diversification': True
                }
            )
        }
