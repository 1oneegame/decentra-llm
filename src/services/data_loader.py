import pandas as pd
import os
from typing import Dict, List
from src.models.client import ClientProfile, Transaction, Transfer

class DataLoader:    
    def __init__(self, data_path: str = 'data/raw/dataset'):
        self.data_path = data_path
        
    def load_client_profiles(self) -> Dict[int, ClientProfile]:
        clients_file = os.path.join(self.data_path, 'clients.csv')
        df = pd.read_csv(clients_file)
        
        profiles = {}
        for _, row in df.iterrows():
            profile = ClientProfile(
                client_code=row['client_code'],
                name=row['name'],
                status=row['status'],
                age=row['age'],
                city=row['city'],
                avg_monthly_balance_KZT=row['avg_monthly_balance_KZT']
            )
            profiles[profile.client_code] = profile
            
        return profiles
    
    def load_client_transactions(self, client_code: int) -> List[Transaction]:
        file_path = os.path.join(self.data_path, f'client_{client_code}_transactions_3m.csv')
        
        if not os.path.exists(file_path):
            return []
            
        df = pd.read_csv(file_path)
        transactions = []
        
        for _, row in df.iterrows():
            transaction = Transaction(
                date=pd.to_datetime(row['date']),
                category=row['category'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transactions.append(transaction)
            
        return transactions
    
    def load_client_transfers(self, client_code: int) -> List[Transfer]:
        file_path = os.path.join(self.data_path, f'client_{client_code}_transfers_3m.csv')
        
        if not os.path.exists(file_path):
            return []
            
        df = pd.read_csv(file_path)
        transfers = []
        
        for _, row in df.iterrows():
            transfer = Transfer(
                date=pd.to_datetime(row['date']),
                type=row['type'],
                direction=row['direction'],
                amount=float(row['amount']),
                currency=row['currency']
            )
            transfers.append(transfer)
            
        return transfers

