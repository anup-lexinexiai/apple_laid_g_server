from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ILlm(ABC):
    @abstractmethod
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LLM request with conversation history, functions, and images
        Returns: {"message": str, "function_call": dict or None}
        """
        pass
    
    @abstractmethod
    def llm_set_instruction(self, instruction: str) -> None:
        """Set system instruction for the LLM"""
        pass
    
    @abstractmethod
    def get_cost(self) -> float:
        """Get the cost for this LLM call"""
        pass