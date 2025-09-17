from openai import OpenAI
from llms.base_llm import BaseLLM
from typing import Dict, Any, List
import json

class LlmGrok(BaseLLM):
    """X.AI Grok models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        # Grok uses developer role for system messages
        self.system_role = "developer" if self.reasoning_model else "system"
    
    def format_function_for_llm(self, func: Dict) -> Dict:
        """Format function for Grok (OpenAI format)"""
        return {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {
                            "type": v.get("type", "string"),
                            "description": v.get("description", "")
                        } for k, v in func.get("parameters", {}).items()
                    },
                    "required": [
                        k for k, v in func.get("parameters", {}).items() 
                        if v.get("required")
                    ]
                }
            }
        }
    
    def process_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Process messages for Grok format"""
        processed = []
        
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            
            if role in ["user", "assistant"]:
                processed.append({"role": role, "content": msg.get("content")})
            elif msg_type == "function_call":
                processed.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": msg.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": msg.get("name"),
                            "arguments": msg.get("arguments")
                        }
                    }]
                })
            elif msg_type == "function_call_output":
                processed.append({
                    "role": "tool",
                    "content": msg.get("output", ""),
                    "tool_call_id": msg.get("call_id")
                })
        
        return processed
    
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Main response method for Grok with web search"""
        try:
            conversation_history = complete_chat.get('conversation_history', [])
            instruction = complete_chat.get('instruction')
            img_data = complete_chat.get('img_data')
            available_functions = complete_chat.get('available_functions', [])
            
            # Process tools
            tools = []
            for func in available_functions:
                # Skip search_internet if native web search enabled
                if self.web_search and func.get("name") == "search_internet":
                    continue
                tools.append(self.format_function_for_llm(func))
            
            # Build messages
            messages = []
            if instruction:
                messages.append({"role": self.system_role, "content": instruction})
            
            messages.extend(self.process_conversation_history(conversation_history))
            
            # Handle images (if vision enabled)
            if img_data and self.vision:
                image_content = [{"type": "text", "text": "Please analyze the attached image(s):"}]
                
                for b64_img in img_data:
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                    })
                
                messages.append({
                    "role": "user",
                    "content": image_content
                })
            
            # Prepare API call parameters
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
            
            # Add web search if enabled
            if self.web_search:
                search_params = {"mode": "auto"}
                kwargs["extra_body"] = {'search_parameters': search_params}
            
            # Make API call
            response = self.client.chat.completions.create(**kwargs)
            
            # Parse response
            return self.parse_response(response)
            
        except Exception as e:
            return {
                "message": f"Grok Error: {str(e)}",
                "function_call": None
            }
    
    def parse_response(self, response) -> Dict[str, Any]:
        """Parse Grok response"""
        choice = response.choices[0]
        message_text = choice.message.content if choice.message.content else ""
        function_call = None
        
        if choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            function_call = {
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "call_id": call.id
            }
        
        result = {"message": message_text}
        if function_call:
            result["function_call"] = function_call
        
        return result or {"message": "No valid response returned from Grok."}