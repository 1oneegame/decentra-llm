from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from src.models.client import ClientProfile, Transaction, Transfer

class ClientAnalyzer:    
    def analyze_client_behavior(self, profile: ClientProfile, 
                              transactions: List[Transaction], 
                              transfers: List[Transfer]) -> Dict[str, Any]:        
        spending_by_category = self._analyze_spending_by_category(transactions)
        
        total_spending = sum(spending_by_category.values())
        foreign_currency_spending = self._calculate_foreign_spending(transactions)
        
        travel_spending = spending_by_category.get('Путешествия', 0) + spending_by_category.get('Отели', 0)
        taxi_spending = spending_by_category.get('Такси', 0)
        restaurant_spending = spending_by_category.get('Кафе и рестораны', 0)
        online_services_spending = (
            spending_by_category.get('Едим дома', 0) +
            spending_by_category.get('Смотрим дома', 0) +
            spending_by_category.get('Играем дома', 0)
        )
        jewelry_cosmetics_spending = (
            spending_by_category.get('Ювелирные украшения', 0) +
            spending_by_category.get('Косметика и Парфюмерия', 0)
        )
        
        transfer_analysis = self._analyze_transfers(transfers)
        
        top_categories = self._get_top_categories(spending_by_category, 3)
        
        monthly_cash_flow = transfer_analysis['monthly_cash_flow']
        savings_potential = max(0, profile.avg_monthly_balance_KZT - total_spending)
        
        risk_indicators = self._calculate_risk_indicators(
            profile, total_spending, monthly_cash_flow, transfer_analysis
        )
        
        return {
            'client_code': profile.client_code,
            'name': profile.name,
            'status': profile.status,
            'age': profile.age,
            'avg_balance': profile.avg_monthly_balance_KZT,
            'spending_by_category': spending_by_category,
            'total_spending': total_spending,
            'foreign_currency_spending': foreign_currency_spending,
            'travel_spending': travel_spending,
            'taxi_spending': taxi_spending,
            'restaurant_spending': restaurant_spending,
            'online_services_spending': online_services_spending,
            'jewelry_cosmetics_spending': jewelry_cosmetics_spending,
            'atm_withdrawals': transfer_analysis['atm_withdrawals'],
            'fx_operations': transfer_analysis['fx_operations'],
            'loan_payments': transfer_analysis['loan_payments'],
            'installment_payments': transfer_analysis['installment_payments'],
            'top_categories': top_categories,
            'monthly_cash_flow': monthly_cash_flow,
            'savings_potential': savings_potential,
            'risk_indicators': risk_indicators
        }
    
    def _analyze_spending_by_category(self, transactions: List[Transaction]) -> Dict[str, float]:
        spending = {}
        for tx in transactions:
            if tx.amount > 0:  
                category = tx.category
                spending[category] = spending.get(category, 0) + tx.amount
        return spending
    
    def _calculate_foreign_spending(self, transactions: List[Transaction]) -> float:
        foreign_spending = 0
        for tx in transactions:
            if tx.currency in ['USD', 'EUR'] and tx.amount > 0:
                rate = 550 if tx.currency == 'USD' else 600
                foreign_spending += tx.amount * rate
        return foreign_spending
    
    def _analyze_transfers(self, transfers: List[Transfer]) -> Dict[str, float]:
        atm_withdrawals = 0
        fx_operations = 0
        loan_payments = 0
        installment_payments = 0
        total_in = 0
        total_out = 0
        
        for tr in transfers:
            if tr.direction == 'in':
                total_in += tr.amount
            else:
                total_out += tr.amount
                
            if tr.type == 'atm_withdrawal':
                atm_withdrawals += tr.amount
            elif tr.type in ['fx_buy', 'fx_sell']:
                fx_operations += tr.amount
            elif tr.type in ['loan_payment_out', 'cc_repayment_out']:
                loan_payments += tr.amount
            elif tr.type == 'installment_payment_out':
                installment_payments += tr.amount
        
        monthly_cash_flow = (total_in - total_out) / 3  
        
        return {
            'atm_withdrawals': atm_withdrawals,
            'fx_operations': fx_operations,
            'loan_payments': loan_payments,
            'installment_payments': installment_payments,
            'monthly_cash_flow': monthly_cash_flow,
            'total_in': total_in,
            'total_out': total_out
        }
    
    def _get_top_categories(self, spending_by_category: Dict[str, float], top_n: int) -> List[str]:
        sorted_categories = sorted(
            spending_by_category.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [cat for cat, amount in sorted_categories[:top_n]]
    
    def _calculate_risk_indicators(self, profile: ClientProfile, total_spending: float, 
                                 monthly_cash_flow: float, transfer_analysis: Dict) -> List[str]:
        indicators = []
        
        if monthly_cash_flow < 0:
            indicators.append('negative_cash_flow')
        
        if profile.avg_monthly_balance_KZT < 100000:
            indicators.append('low_balance')
        
        if transfer_analysis['loan_payments'] > profile.avg_monthly_balance_KZT * 0.3:
            indicators.append('high_debt_burden')
        
        if total_spending > profile.avg_monthly_balance_KZT * 2:
            indicators.append('high_spending_ratio')
        
        return indicators
    
    def create_behavioral_features(self, transactions: List[Transaction], transfers: List[Transfer]) -> Dict[str, float]:        
        if not transactions:
            return {
                'avg_transaction_amount': 0,
                'transaction_frequency': 0,
                'transaction_volatility': 0,
                'unique_categories': 0,
                'evening_spending_ratio': 0,
                'weekend_spending_ratio': 0,
                'large_transaction_ratio': 0,
                'foreign_currency_ratio': 0,
                'incoming_to_outgoing_ratio': 0,
                'net_transfer_ratio': 0,
                'transfer_frequency': 0
            }
        
        amounts = [tx.amount for tx in transactions if tx.amount > 0]
        avg_transaction_amount = np.mean(amounts) if amounts else 0
        transaction_frequency = len(transactions) / 90  # за 3 месяца
        transaction_volatility = np.std(amounts) if len(amounts) > 1 else 0
        unique_categories = len(set(tx.category for tx in transactions))
        
        total_spending = sum(amounts)
        evening_spending = sum(tx.amount for tx in transactions if hash(str(tx.date)) % 24 >= 18)
        weekend_spending = sum(tx.amount for tx in transactions if hash(str(tx.date)) % 7 >= 5)
        
        evening_spending_ratio = evening_spending / max(total_spending, 1)
        weekend_spending_ratio = weekend_spending / max(total_spending, 1)
        
        large_transactions = [amt for amt in amounts if avg_transaction_amount > 0 and amt > avg_transaction_amount * 2]
        large_transaction_ratio = len(large_transactions) / max(len(amounts), 1) if amounts else 0
        
        foreign_transactions = [tx for tx in transactions if tx.currency in ['USD', 'EUR']]
        foreign_currency_ratio = len(foreign_transactions) / max(len(transactions), 1)
        
        if transfers:
            incoming = [tr.amount for tr in transfers if tr.direction == 'in']
            outgoing = [tr.amount for tr in transfers if tr.direction == 'out']
            
            total_incoming = sum(incoming)
            total_outgoing = sum(outgoing)
            
            incoming_to_outgoing_ratio = total_incoming / max(total_outgoing, 1)
            net_transfer_ratio = (total_incoming - total_outgoing) / max(total_incoming + total_outgoing, 1)
            transfer_frequency = len(transfers) / 90
        else:
            incoming_to_outgoing_ratio = 0
            net_transfer_ratio = 0
            transfer_frequency = 0
        
        return {
            'avg_transaction_amount': avg_transaction_amount,
            'transaction_frequency': transaction_frequency,
            'transaction_volatility': transaction_volatility,
            'unique_categories': unique_categories,
            'evening_spending_ratio': evening_spending_ratio,
            'weekend_spending_ratio': weekend_spending_ratio,
            'large_transaction_ratio': large_transaction_ratio,
            'foreign_currency_ratio': foreign_currency_ratio,
            'incoming_to_outgoing_ratio': incoming_to_outgoing_ratio,
            'net_transfer_ratio': net_transfer_ratio,
            'transfer_frequency': transfer_frequency
        }
