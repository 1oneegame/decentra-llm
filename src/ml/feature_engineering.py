import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Tuple
from src.models.client import ClientAnalysis
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.selected_features = None
        
    def create_features(self, client_analysis) -> Dict[str, float]:
        
        def safe_get(key, default=0):
            value = client_analysis.get(key, default)
            return value if value is not None else default
        
        features = {
            'age': safe_get('age', 35),
            'avg_balance': safe_get('avg_balance', 0),
            'total_spending': safe_get('total_spending', 0),
            'savings_potential': safe_get('savings_potential', 0),
            'monthly_cash_flow': safe_get('monthly_cash_flow', 0),
            
            
            'spending_to_balance_ratio': safe_get('total_spending') / max(safe_get('avg_balance'), 1),
            'savings_ratio': safe_get('savings_potential') / max(safe_get('avg_balance'), 1),
            'cash_flow_ratio': safe_get('monthly_cash_flow') / max(safe_get('avg_balance'), 1),
            
            
            'travel_spending_ratio': safe_get('travel_spending') / max(safe_get('total_spending'), 1),
            'taxi_spending_ratio': safe_get('taxi_spending') / max(safe_get('total_spending'), 1),
            'restaurant_spending_ratio': safe_get('restaurant_spending') / max(safe_get('total_spending'), 1),
            'online_services_ratio': safe_get('online_services_spending') / max(safe_get('total_spending'), 1),
            'jewelry_cosmetics_ratio': safe_get('jewelry_cosmetics_spending') / max(safe_get('total_spending'), 1),
            
            
            'atm_usage_ratio': safe_get('atm_withdrawals') / max(safe_get('avg_balance'), 1),
            'fx_activity_ratio': safe_get('fx_operations') / max(safe_get('total_spending'), 1),
            'loan_burden_ratio': safe_get('loan_payments') / max(safe_get('avg_balance'), 1),
            
            
            'risk_score': len(safe_get('risk_indicators', [])),
            'has_negative_cash_flow': 1 if 'negative_cash_flow' in safe_get('risk_indicators', []) else 0,
            'has_low_balance': 1 if 'low_balance' in safe_get('risk_indicators', []) else 0,
            'has_high_debt': 1 if 'high_debt_burden' in safe_get('risk_indicators', []) else 0,
            
            
            'spending_diversity': len([cat for cat, amount in safe_get('spending_by_category', {}).items() if amount > 0]),
            'top_category_dominance': max(safe_get('spending_by_category', {}).values()) / max(safe_get('total_spending'), 1) if safe_get('spending_by_category', {}) else 0,
            
            
            'is_premium': 1 if safe_get('status', '') == 'Премиальный клиент' else 0,
            'is_salary': 1 if safe_get('status', '') == 'Зарплатный клиент' else 0,
            'is_student': 1 if safe_get('status', '') == 'Студент' else 0,
            'is_standard': 1 if safe_get('status', '') == 'Стандартный клиент' else 0,
            
            
            'is_young': 1 if safe_get('age', 35) <= 25 else 0,
            'is_middle_age': 1 if 25 < safe_get('age', 35) <= 45 else 0,
            'is_mature': 1 if safe_get('age', 35) > 45 else 0,
            
            
            'travel_enthusiast': 1 if (safe_get('travel_spending') + safe_get('taxi_spending')) / max(safe_get('total_spending'), 1) > 0.15 else 0,
            'premium_lifestyle': 1 if (safe_get('restaurant_spending') / max(safe_get('total_spending'), 1) > 0.08 and safe_get('avg_balance') > 1000000) else 0,
            'digital_native': 1 if safe_get('online_services_spending') / max(safe_get('total_spending'), 1) > 0.10 else 0,
            'high_net_worth': 1 if safe_get('avg_balance') > 2000000 else 0,
            'active_trader': 1 if safe_get('fx_operations') / max(safe_get('total_spending'), 1) > 0.05 else 0,
            'conservative_saver': 1 if (safe_get('avg_balance') / max(safe_get('total_spending'), 1) > 10) else 0,
            'young_investor': 1 if (safe_get('age', 35) < 35 and safe_get('avg_balance') > 300000) else 0,
            'cash_strapped': 1 if (safe_get('avg_balance') < 100000 or 'negative_cash_flow' in safe_get('risk_indicators', [])) else 0,
        }
        
        advanced_features = self._create_advanced_features(client_analysis)
        features.update(advanced_features)
        
        return features
    
    def _create_advanced_features(self, analysis: Dict) -> Dict[str, float]:
        total_spending = max(analysis.get('total_spending', 1), 1)
        balance = analysis.get('avg_balance', 0)
        
        features = {}
        
        features.update({
            'spending_momentum': self._calculate_spending_momentum(analysis),
            'financial_stability': self._calculate_financial_stability(analysis),
            'lifestyle_score': self._calculate_lifestyle_score(analysis),
            'investment_readiness': self._calculate_investment_readiness(analysis),
            'risk_tolerance': self._calculate_risk_tolerance(analysis),
            'seasonal_spending_variance': self._calculate_seasonal_variance(analysis),
            'category_concentration': self._calculate_category_concentration(analysis),
            'wealth_percentile': self._estimate_wealth_percentile(analysis),
            'spending_efficiency': balance / max(total_spending, 1) if total_spending > 0 else 0,
            'financial_growth_potential': self._calculate_growth_potential(analysis)
        })
        
        interaction_features = self._create_interaction_features(analysis)
        features.update(interaction_features)
        
        return features
    
    def _calculate_spending_momentum(self, analysis: Dict) -> float:
        recent_spending = analysis.get('total_spending', 0) or 0
        avg_balance = analysis.get('avg_balance', 0) or 0
        monthly_cash_flow = analysis.get('monthly_cash_flow', 0) or 0
        
        if avg_balance > 0:
            momentum = (recent_spending + monthly_cash_flow) / avg_balance
            return min(momentum, 2.0)
        return 0.0
    
    def _calculate_financial_stability(self, analysis: Dict) -> float:
        risk_indicators = len(analysis.get('risk_indicators', []) or [])
        balance = analysis.get('avg_balance', 0) or 0
        cash_flow = analysis.get('monthly_cash_flow', 0) or 0
        
        stability = 1.0
        stability -= risk_indicators * 0.2
        stability += min(balance / 1000000, 0.5) if balance > 0 else 0
        stability += min(cash_flow / 100000, 0.3) if cash_flow > 0 else (cash_flow / 100000 * 0.5 if cash_flow != 0 else 0)
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_lifestyle_score(self, analysis: Dict) -> float:
        restaurant = analysis.get('restaurant_spending', 0) or 0
        travel = analysis.get('travel_spending', 0) or 0
        entertainment = analysis.get('entertainment_spending', 0) or 0
        jewelry = analysis.get('jewelry_cosmetics_spending', 0) or 0
        total = analysis.get('total_spending', 0) or 0
        
        if total <= 0:
            return 0.0
        
        lifestyle_ratio = (restaurant + travel + entertainment + jewelry) / total
        return min(lifestyle_ratio * 2, 1.0)
    
    def _calculate_investment_readiness(self, analysis: Dict) -> float:
        age = analysis.get('age', 35)
        balance = analysis.get('avg_balance', 0)
        risk_indicators = len(analysis.get('risk_indicators', []))
        stability = self._calculate_financial_stability(analysis)
        
        readiness = 0.0
        readiness += min((40 - age) / 40, 0.3) if age < 40 else 0.1
        readiness += min(balance / 500000, 0.4)
        readiness += stability * 0.3
        readiness -= risk_indicators * 0.1
        
        return max(0.0, min(1.0, readiness))
    
    def _calculate_risk_tolerance(self, analysis: Dict) -> float:
        age = analysis.get('age', 35) or 35
        balance = analysis.get('avg_balance', 0) or 0
        total_spending = analysis.get('total_spending', 0) or 0
        spending_ratio = total_spending / max(balance, 1) if balance > 0 else 0
        
        tolerance = 0.5
        tolerance += (50 - age) / 100 if age < 50 else -(age - 50) / 200
        tolerance += min(balance / 1000000, 0.3) if balance > 0 else 0
        tolerance -= min(spending_ratio, 0.4)
        
        return max(0.0, min(1.0, tolerance))
    
    def _calculate_seasonal_variance(self, analysis: Dict) -> float:
        spending_categories = analysis.get('spending_by_category', {})
        if not spending_categories:
            return 0.0
        
        values = list(spending_categories.values())
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        variance = np.var(values) / max(mean_val, 1) if mean_val > 0 else 0
        return min(variance, 2.0)
    
    def _calculate_category_concentration(self, analysis: Dict) -> float:
        spending_categories = analysis.get('spending_by_category', {})
        if not spending_categories:
            return 1.0
        
        values = np.array(list(spending_categories.values()))
        total = np.sum(values)
        
        if total == 0:
            return 1.0
        
        proportions = values / total
        hhi = np.sum(proportions ** 2)
        return hhi
    
    def _estimate_wealth_percentile(self, analysis: Dict) -> float:
        balance = analysis.get('avg_balance', 0)
        
        if balance < 100000:
            return 0.2
        elif balance < 500000:
            return 0.4
        elif balance < 1000000:
            return 0.6
        elif balance < 5000000:
            return 0.8
        else:
            return 0.95
    
    def _calculate_growth_potential(self, analysis: Dict) -> float:
        age = analysis.get('age', 35)
        balance = analysis.get('avg_balance', 0)
        cash_flow = analysis.get('monthly_cash_flow', 0)
        stability = self._calculate_financial_stability(analysis)
        
        potential = 0.0
        potential += (65 - age) / 65 * 0.3 if age < 65 else 0.1
        potential += min(cash_flow / 200000, 0.4) if cash_flow > 0 else 0.0
        potential += stability * 0.3
        
        return max(0.0, min(1.0, potential))
    
    def _create_interaction_features(self, analysis: Dict) -> Dict[str, float]:
        features = {}
        
        age = analysis.get('age', 35) or 35
        balance = analysis.get('avg_balance', 0) or 0
        total_spending = analysis.get('total_spending', 0) or 0
        
        features['age_balance_interaction'] = age * balance / 1000000 if balance > 0 else 0
        features['age_spending_interaction'] = age * total_spending / 100000 if total_spending > 0 else 0
        features['balance_spending_interaction'] = balance * total_spending / 1000000000 if balance > 0 and total_spending > 0 else 0
        
        restaurant_spending = analysis.get('restaurant_spending', 0) or 0
        travel_spending = analysis.get('travel_spending', 0) or 0
        taxi_spending = analysis.get('taxi_spending', 0) or 0
        
        restaurant_ratio = restaurant_spending / max(total_spending, 1) if total_spending > 0 else 0
        travel_ratio = (travel_spending + taxi_spending) / max(total_spending, 1) if total_spending > 0 else 0
        
        features['lifestyle_wealth_interaction'] = (restaurant_ratio + travel_ratio) * balance / 1000000 if balance > 0 else 0
        features['risk_age_interaction'] = len(analysis.get('risk_indicators', []) or []) * age
        
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
            
            mean_amount = tx_df['amount'].mean()
            std_amount = tx_df['amount'].std()
            sum_amount = tx_df['amount'].sum()
            
            features.update({
                'avg_transaction_amount': mean_amount if not pd.isna(mean_amount) else 0,
                'transaction_frequency': len(tx_df) / 90,
                'weekend_spending_ratio': tx_df[tx_df['is_weekend'] == 1]['amount'].sum() / max(sum_amount, 1),
                'evening_spending_ratio': tx_df[tx_df['hour'] >= 18]['amount'].sum() / max(sum_amount, 1),
                'transaction_volatility': (std_amount / max(mean_amount, 1)) if not pd.isna(std_amount) and mean_amount > 0 else 0,
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
