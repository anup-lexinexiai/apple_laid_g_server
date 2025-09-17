from typing import Dict, Any
from llms.llm_factory import create_llm
from config import config

class LLMRouter:
    """Routes LLM requests to appropriate handler"""
    
    def __init__(self):
        self.llm_instances = {}
    
    def get_llm_instance(self, model_name: str):
        """Get or create LLM instance (cached)"""
        if model_name not in self.llm_instances:
            self.llm_instances[model_name] = create_llm(model_name)
        return self.llm_instances[model_name]
    
    def process_request(self, model_name: str, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM request"""
        llm = self.get_llm_instance(model_name)
        
        # Set instruction if provided
        instruction = complete_chat.get('instruction')
        if instruction:
            llm.llm_set_instruction(instruction)
        
        # Process and return response
        return llm.llm_response(complete_chat)
    
    def get_model_cost(self, model_name: str) -> float:
        """Get cost for a model"""
        model_config = config.get_llm_config(model_name)
        return model_config.get('cost', 0) if model_config else 0