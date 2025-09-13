import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
from src.models.client import ClientAnalysis

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, client_analysis) -> Dict[str, float]:
        
        features = {
            'age': client_analysis['age'],
            'avg_balance': client_analysis['avg_balance'],
            'total_spending': client_analysis['total_spending'],
            'savings_potential': client_analysis['savings_potential'],
            'monthly_cash_flow': client_analysis['monthly_cash_flow'],
            
            
            'spending_to_balance_ratio': client_analysis['total_spending'] / max(client_analysis['avg_balance'], 1),
            'savings_ratio': client_analysis['savings_potential'] / max(client_analysis['avg_balance'], 1),
            'cash_flow_ratio': client_analysis['monthly_cash_flow'] / max(client_analysis['avg_balance'], 1),
            
            
            'travel_spending_ratio': client_analysis['travel_spending'] / max(client_analysis['total_spending'], 1),
            'taxi_spending_ratio': client_analysis['taxi_spending'] / max(client_analysis['total_spending'], 1),
            'restaurant_spending_ratio': client_analysis['restaurant_spending'] / max(client_analysis['total_spending'], 1),
            'online_services_ratio': client_analysis['online_services_spending'] / max(client_analysis['total_spending'], 1),
            'jewelry_cosmetics_ratio': client_analysis['jewelry_cosmetics_spending'] / max(client_analysis['total_spending'], 1),
            
            
            'atm_usage_ratio': client_analysis['atm_withdrawals'] / max(client_analysis['avg_balance'], 1),
            'fx_activity_ratio': client_analysis['fx_operations'] / max(client_analysis['total_spending'], 1),
            'loan_burden_ratio': client_analysis['loan_payments'] / max(client_analysis['avg_balance'], 1),
            
            
            'risk_score': len(client_analysis['risk_indicators']),
            'has_negative_cash_flow': 1 if 'negative_cash_flow' in client_analysis['risk_indicators'] else 0,
            'has_low_balance': 1 if 'low_balance' in client_analysis['risk_indicators'] else 0,
            'has_high_debt': 1 if 'high_debt_burden' in client_analysis['risk_indicators'] else 0,
            
            
            'spending_diversity': len([cat for cat, amount in client_analysis['spending_by_category'].items() if amount > 0]),
            'top_category_dominance': max(client_analysis['spending_by_category'].values()) / max(client_analysis['total_spending'], 1) if client_analysis['spending_by_category'] else 0,
            
            
            'is_premium': 1 if client_analysis['status'] == 'Премиальный клиент' else 0,
            'is_salary': 1 if client_analysis['status'] == 'Зарплатный клиент' else 0,
            'is_student': 1 if client_analysis['status'] == 'Студент' else 0,
            'is_standard': 1 if client_analysis['status'] == 'Стандартный клиент' else 0,
            
            
            'is_young': 1 if client_analysis['age'] <= 25 else 0,
            'is_middle_age': 1 if 25 < client_analysis['age'] <= 45 else 0,
            'is_mature': 1 if client_analysis['age'] > 45 else 0,
            
            
            'travel_enthusiast': 1 if (client_analysis['travel_spending'] + client_analysis['taxi_spending']) / max(client_analysis['total_spending'], 1) > 0.15 else 0,
            'premium_lifestyle': 1 if (client_analysis['restaurant_spending'] / max(client_analysis['total_spending'], 1) > 0.08 and client_analysis['avg_balance'] > 1000000) else 0,
            'digital_native': 1 if client_analysis['online_services_spending'] / max(client_analysis['total_spending'], 1) > 0.10 else 0,
            'high_net_worth': 1 if client_analysis['avg_balance'] > 2000000 else 0,
            'active_trader': 1 if client_analysis['fx_operations'] / max(client_analysis['total_spending'], 1) > 0.05 else 0,
            'conservative_saver': 1 if (client_analysis['avg_balance'] / max(client_analysis['total_spending'], 1) > 10) else 0,
            'young_investor': 1 if (client_analysis['age'] < 35 and client_analysis['avg_balance'] > 300000) else 0,
            'cash_strapped': 1 if (client_analysis['avg_balance'] < 100000 or 'negative_cash_flow' in client_analysis['risk_indicators']) else 0,
        }
        
        return features
    
    def create_behavioral_features(self, transactions: List, transfers: List) -> Dict[str, float]:
        
        features = {}
        
        if transactions:
            tx_df = pd.DataFrame([{
                'amount': tx.amount,
                'category': tx.category,
                'date': tx.date,
                'currency': tx.currency
            } for tx in transactions])
            
            
            tx_df['hour'] = tx_df['date'].dt.hour
            tx_df['day_of_week'] = tx_df['date'].dt.dayofweek
            tx_df['is_weekend'] = tx_df['day_of_week'].isin([5, 6]).astype(int)
            
            features.update({
                'avg_transaction_amount': tx_df['amount'].mean(),
                'transaction_frequency': len(tx_df) / 90,  
                'weekend_spending_ratio': tx_df[tx_df['is_weekend'] == 1]['amount'].sum() / max(tx_df['amount'].sum(), 1),
                'evening_spending_ratio': tx_df[tx_df['hour'] >= 18]['amount'].sum() / max(tx_df['amount'].sum(), 1),
                'transaction_volatility': tx_df['amount'].std() / max(tx_df['amount'].mean(), 1),
                'unique_categories': tx_df['category'].nunique(),
            })
        
        if transfers:
            tr_df = pd.DataFrame([{
                'amount': tr.amount,
                'type': tr.type,
                'direction': tr.direction,
                'date': tr.date
            } for tr in transfers])
            
            
            incoming = tr_df[tr_df['direction'] == 'in']['amount'].sum()
            outgoing = tr_df[tr_df['direction'] == 'out']['amount'].sum()
            
            features.update({
                'transfer_frequency': len(tr_df) / 90,
                'net_transfer_ratio': (incoming - outgoing) / max(incoming + outgoing, 1),
                'incoming_to_outgoing_ratio': incoming / max(outgoing, 1),
                'transfer_types_diversity': tr_df['type'].nunique(),
            })
        
        return features
    
    def prepare_dataset(self, clients_features: List[Dict]) -> Tuple[np.ndarray, List[str]]:

        df = pd.DataFrame(clients_features)
        
        df = df.fillna(0)
        
        feature_names = df.columns.tolist()
        X = self.scaler.fit_transform(df)
        
        return X, feature_names
