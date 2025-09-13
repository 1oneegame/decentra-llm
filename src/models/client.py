from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class ClientProfile:
    client_code: int
    name: str
    status: str
    age: int
    city: str
    avg_monthly_balance_KZT: float

@dataclass  
class Transaction:
    date: datetime
    category: str
    amount: float
    currency: str

@dataclass
class Transfer:
    date: datetime
    type: str
    direction: str 
    amount: float
    currency: str

@dataclass
class ClientAnalysis:
    client_code: int
    name: str
    status: str
    age: int
    avg_balance: float
    spending_by_category: Dict[str, float]
    total_spending: float
    foreign_currency_spending: float
    travel_spending: float
    online_services_spending: float
    taxi_spending: float
    restaurant_spending: float
    jewelry_cosmetics_spending: float
    atm_withdrawals: float
    fx_operations: float
    loan_payments: float
    installment_payments: float
    top_categories: List[str]
    monthly_cash_flow: float
    savings_potential: float
    risk_indicators: List[str]
