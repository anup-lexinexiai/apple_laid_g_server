import os
import yaml
from typing import Dict, Any

class Config:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self):
        """Load config based on environment"""
        env = os.getenv('ENVIRONMENT', 'local')
        config_path = f'configs/config.{env}.yaml'
        
        if not os.path.exists(config_path):
            config_path = 'configs/config.local.yaml'
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get config value by dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_llm_config(self, model_name: str) -> Dict[str, Any]:
        """Get specific LLM model config"""
        return self.get(f'llm_models.{model_name}', {})
    
    def get_all_llm_models(self) -> Dict[str, Any]:
        """Get all LLM models config"""
        return self.get('llm_models', {})

# Singleton instance
config = Config()