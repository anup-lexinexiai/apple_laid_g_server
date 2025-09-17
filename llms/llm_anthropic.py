from anthropic import Anthropic
from llms.base_llm import BaseLLM
from typing import Dict, Any, List, Optional
import json

class LlmAnthropic(BaseLLM):
    """Anthropic Claude models including Haiku, Sonnet, Opus"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic(api_key=self.api_key)
        
        # Claude-specific config
        self.web_search_max_uses = self.model_config.get('web_search_max_uses', 5)
    
    def format_function_for_llm(self, func: Dict) -> Dict:
        """Format function for Claude tools API"""
        return {
            "type": "custom",
            "name": func["name"],
            "description": func["description"],
            "input_schema": {
                "type": "object",
                "properties": {
                    k: {
                        "type": v.get("type", "string"),
                        "description": v.get("description", "")
                    }
                    for k, v in func.get("parameters", {}).items()
                },
                "required": [
                    k for k, v in func.get("parameters", {}).items() 
                    if v.get("required")
                ]
            }
        }
    
    def process_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Process messages for Claude format"""
        processed = []
        
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            
            if role in ["user", "assistant"]:
                processed.append({
                    "role": role,
                    "content": [{"type": "text", "text": msg.get("content", "")}]
                })
            elif msg_type == "function_call":
                processed.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": msg.get("name"),
                        "id": msg.get("call_id"),
                        "input": json.loads(msg.get("arguments", "{}"))
                    }]
                })
            elif msg_type == "function_call_output":
                processed.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("call_id"),
                        "content": msg.get("output", "")
                    }]
                })
        
        return processed
    
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Main response method with Claude capabilities"""
        try:
            conversation_history = complete_chat.get('conversation_history', [])
            instruction = complete_chat.get('instruction', '')
            img_data = complete_chat.get('img_data')
            available_functions = complete_chat.get('available_functions', [])
            
            # Process tools
            tools = []
            for func in available_functions:
                if self.web_search and func.get("name") == "search_internet":
                    continue
                tools.append(self.format_function_for_llm(func))
            
            # Add web search if enabled
            if self.web_search:
                tools.append({
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": self.web_search_max_uses
                })
            
            # Process messages
            messages = self.process_conversation_history(conversation_history)
            
            # Handle images if present
            if img_data:
                for b64_img in img_data:
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_img
                            }
                        }]
                    })
            
            # Make API call
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
            
            if instruction:
                kwargs["system"] = instruction
            
            # Add thinking for reasoning models
            if self.reasoning_model:
                try:
                    response = self.client.messages.create(
                        **kwargs,
                        thinking={"type": "enabled"}
                    )
                except:
                    # Fallback without thinking
                    response = self.client.messages.create(**kwargs)
            else:
                response = self.client.messages.create(**kwargs)
            
            # Parse response
            return self.parse_response(response)
            
        except Exception as e:
            return {
                "message": f"Claude Error: {str(e)}",
                "function_call": None
            }
    
    def parse_response(self, response) -> Dict[str, Any]:
        """Parse Claude response"""
        message_text = ""
        function_call = None
        
        for block in response.content:
            if block.type == "text":
                message_text += block.text
            elif block.type == "tool_use":
                function_call = {
                    "name": block.name,
                    "arguments": block.input,
                    "call_id": block.id
                }
        
        result = {}
        if message_text:
            result["message"] = message_text
        if function_call:
            result["function_call"] = function_call
        
        return result