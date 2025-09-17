import os
from app.create_app import create_app

# Set environment
env = os.getenv('ENVIRONMENT', 'local')

app = create_app()

if __name__ == '__main__':
    if env == 'local':
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        app.run(host='0.0.0.0', port=8080)