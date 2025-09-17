from functools import wraps
from flask import request, jsonify
import jwt
import requests
import json
from jwt.algorithms import RSAAlgorithm
from cachetools import TTLCache
from config import config

# Cache for Apple's public keys (valid for 1 hour)
APPLE_PUBLIC_KEYS_CACHE = TTLCache(maxsize=1, ttl=3600)

def get_apple_public_keys():
    """Fetch and cache Apple's public keys for JWT verification."""
    if 'keys' in APPLE_PUBLIC_KEYS_CACHE:
        return APPLE_PUBLIC_KEYS_CACHE['keys']
    
    try:
        response = requests.get("https://appleid.apple.com/auth/keys")
        response.raise_for_status()
        keys = response.json().get("keys", [])
        APPLE_PUBLIC_KEYS_CACHE['keys'] = keys
        return keys
    except requests.exceptions.RequestException as e:
        # Log the error appropriately in a real application
        print(f"Error fetching Apple public keys: {e}")
        raise ValueError("Failed to fetch Apple public keys") from e

def verify_apple_token(identity_token: str):
    """
    Verifies the Apple ID identity token and returns the decoded token.
    """
    apple_keys = get_apple_public_keys()
    
    try:
        header = jwt.get_unverified_header(identity_token)
    except jwt.DecodeError as e:
        raise ValueError(f"Invalid token header: {e}")

    key = next((k for k in apple_keys if k.get("kid") == header.get("kid")), None)
    if not key:
        raise ValueError("Invalid identity token: Key ID not found in Apple's public keys.")
    
    try:
        public_key = RSAAlgorithm.from_jwk(json.dumps(key))
        
        decoded_token = jwt.decode(
            identity_token,
            key=public_key,
            algorithms=["RS256"],
            audience=config.get('apple.client_id'),
            issuer="https://appleid.apple.com"
        )
        return decoded_token
        
    except jwt.ExpiredSignatureError:
        raise ValueError("Expired identity token")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid identity token: {e}")

def require_auth(f):
    """Decorator to enforce authentication on API routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized', 'message': 'Authorization header is missing or invalid.'}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        # For local development, allow a test token
        if config.get('environment') == 'local' and token == 'test-token':
            request.user_id = 'test-user'
            return f(*args, **kwargs)
        
        # Verify the real token for non-local environments
        try:
            decoded = verify_apple_token(token)
            user_id = decoded.get('sub')
            if not user_id:
                return jsonify({'error': 'Invalid token', 'message': 'Token is missing subject claim.'}), 401
            
            request.user_id = user_id
            return f(*args, **kwargs)
            
        except ValueError as e:
            return jsonify({'error': 'Invalid token', 'message': str(e)}), 401
        except Exception as e:
            # Catch any other unexpected errors during token validation
            print(f"Unexpected error during authentication: {e}")
            return jsonify({'error': 'Authentication failed', 'message': 'An unexpected error occurred.'}), 500
            
    return decorated_function
