from datetime import datetime
from services.firestore_service import get_db
from config import config

class CreditService:
    def __init__(self):
        self.db = get_db()
        self.free_credits = config.get('app.free_credits', 10)
    
    def get_balance(self, user_id):
        """Get user's credit balance"""
        doc = self.db.collection('credits').document(user_id).get()
        
        if not doc.exists:
            # New user, give free credits
            self.initialize_user(user_id)
            return self.free_credits
        
        return doc.to_dict().get('balance', 0)
    
    def initialize_user(self, user_id):
        """Initialize new user with free credits"""
        self.db.collection('credits').document(user_id).set({
            'balance': self.free_credits,
            'total_used': 0,
            'created_at': datetime.utcnow(),
            'last_updated': datetime.utcnow()
        })
    
    def has_sufficient_credits(self, user_id, amount):
        """Check if user has enough credits"""
        balance = self.get_balance(user_id)
        return balance >= amount
    
    def deduct_credits(self, user_id, amount, model_name):
        """Deduct credits and log transaction"""
        # Update balance
        credit_ref = self.db.collection('credits').document(user_id)
        credit_doc = credit_ref.get()
        
        if not credit_doc.exists:
            self.initialize_user(user_id)
            credit_doc = credit_ref.get()
        
        current_data = credit_doc.to_dict()
        new_balance = current_data['balance'] - amount
        
        credit_ref.update({
            'balance': new_balance,
            'total_used': current_data.get('total_used', 0) + amount,
            'last_updated': datetime.utcnow()
        })
        
        # Log transaction
        self.db.collection('transactions').add({
            'user_id': user_id,
            'model': model_name,
            'credits_used': amount,
            'timestamp': datetime.utcnow(),
            'balance_after': new_balance
        })
        
        return new_balance
    
    def add_credits(self, user_id, amount):
        """Add credits to user account"""
        credit_ref = self.db.collection('credits').document(user_id)
        credit_doc = credit_ref.get()
        
        if not credit_doc.exists:
            self.initialize_user(user_id)
            credit_doc = credit_ref.get()
        
        current_balance = credit_doc.to_dict()['balance']
        new_balance = current_balance + amount
        
        credit_ref.update({
            'balance': new_balance,
            'last_updated': datetime.utcnow()
        })
        
        # Log credit addition
        self.db.collection('transactions').add({
            'user_id': user_id,
            'type': 'credit_added',
            'credits_added': amount,
            'timestamp': datetime.utcnow(),
            'balance_after': new_balance
        })
        
        return new_balance