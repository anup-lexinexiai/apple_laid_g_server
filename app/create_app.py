from flask import Flask, jsonify
from flask_cors import CORS
from config import config

def create_app():
    app = Flask(__name__)
    
    # Setup CORS
    env = config.get('environment', 'local')
    if env == 'local':
        CORS(app, origins="*")
    else:
        CORS(app, origins=["https://yourdomain.com"])
    
    # Initialize Firebase
    from services.firestore_service import init_firestore
    init_firestore()
    
    # Register routes
    from app.api import register_routes
    register_routes(app)
    
    # Health check
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'environment': env
        })
    
    return app