from google.cloud import firestore
from config import config

db = None

def init_firestore():
    global db
    project_id = config.get('firestore.project_id')
    
    if config.get('environment') == 'local':
        # For local, you can use emulator or real project
        db = firestore.Client(project=project_id)
    else:
        db = firestore.Client(project=project_id)
    
    return db

def get_db():
    global db
    if db is None:
        init_firestore()
    return db