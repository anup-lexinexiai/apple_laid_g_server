from functools import wraps
from flask import request, jsonify
import jwt
from config import config

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        # For local development, accept test token
        if config.get('environment') == 'local' and token == 'test-token':
            request.user_id = 'test-user'
            return f(*args, **kwargs)
        
        # Verify real token
        try:
            # Implement Apple Sign-In verification here
            decoded = verify_apple_token(token)
            request.user_id = decoded['sub']
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
    
    return decorated_function

def verify_apple_token(token):
    # Implement Apple token verification
    # For now, return mock data
    return {'sub': 'user-123'}