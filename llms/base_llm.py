from abc import ABC
from interfaces.i_llm import ILlm
from config import config
from typing import Dict, Any, List, Optional
import json

class BaseLLM(ILlm):
    """Base class with common LLM functionality"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.model_config = config.get_llm_config(model_name)
        
        if not self.model_config:
            raise ValueError(f"No configuration found for model: {model_name}")
        
        # Common config
        self.api_key = self.model_config.get('api_key')
        self.model = self.model_config.get('model')
        self.temperature = self.model_config.get('temperature', 0)
        self.max_tokens = self.model_config.get('max_tokens', 4000)
        self.cost = self.model_config.get('cost', 0)
        
        # Special flags
        self.reasoning_model = self.model_config.get('reasoning_model', False)
        self.web_search = self.model_config.get('web_search', False)
        self.vision = self.model_config.get('vision', False)
        self.priority = self.model_config.get('priority', False)
        
        self.instruction = None
    
    def get_cost(self) -> float:
        return self.cost
    
    def llm_set_instruction(self, instruction: str) -> None:
        self.instruction = instruction
    
    def process_functions(self, available_functions: List[Dict]) -> List[Dict]:
        """Process available functions into tool format"""
        tools = []
        
        for func in available_functions:
            # Skip web search if native web search is enabled
            if self.web_search and func.get("name") == "search_internet":
                continue
            
            tool = self.format_function_for_llm(func)
            if tool:
                tools.append(tool)
        
        return tools
    
    def format_function_for_llm(self, func: Dict) -> Optional[Dict]:
        """Override in child classes for specific formatting"""
        return func
    
    def process_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Process conversation history including function calls"""
        processed = []
        
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            
            if msg_type == "function_call":
                processed.append(self.format_function_call(msg))
            elif msg_type == "function_call_output":
                processed.append(self.format_function_output(msg))
            elif role in ["user", "assistant", "system"]:
                processed.append(msg)
        
        return processed
    
    def format_function_call(self, msg: Dict) -> Dict:
        """Format function call for specific LLM - override in child"""
        return msg
    
    def format_function_output(self, msg: Dict) -> Dict:
        """Format function output for specific LLM - override in child"""
        return msg