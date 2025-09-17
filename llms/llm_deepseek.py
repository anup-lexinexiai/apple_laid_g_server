from openai import OpenAI
from llms.base_llm import BaseLLM
from typing import Dict, Any, List
import json

class LlmDeepSeek(BaseLLM):
    """DeepSeek models (OpenAI-compatible)"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.system_role = "system"
    
    def format_function_for_llm(self, func: Dict) -> Dict:
        """Format function for DeepSeek (OpenAI format)"""
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
        """Process messages for DeepSeek format"""
        processed = []
        
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            
            if role in ["user", "assistant", "system", "tool"]:
                processed.append(msg)
            elif msg_type == "function_call":
                processed.append({
                    "role": "assistant",
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
                    "tool_call_id": msg.get("call_id"),
                    "content": msg.get("output", "")
                })
        
        return processed
    
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Main response method for DeepSeek"""
        try:
            conversation_history = complete_chat.get('conversation_history', [])
            instruction = complete_chat.get('instruction')
            img_data = complete_chat.get('img_data')
            available_functions = complete_chat.get('available_functions', [])
            
            # Process tools
            tools = []
            for func in available_functions:
                tools.append(self.format_function_for_llm(func))
            
            # Build messages
            messages = []
            if instruction:
                messages.append({"role": self.system_role, "content": instruction})
            
            # Process conversation history
            messages.extend(self.process_conversation_history(conversation_history))
            
            # Handle images
            if img_data:
                # Add image as user message
                image_content = []
                image_content.append({"type": "text", "text": "Please analyze the image(s):"})
                
                for b64_img in img_data:
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                    })
                
                messages.append({
                    "role": "user",
                    "content": image_content
                })
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response
            return self.parse_response(response)
            
        except Exception as e:
            return {
                "message": f"DeepSeek Error: {str(e)}",
                "function_call": None
            }
    
    def parse_response(self, response) -> Dict[str, Any]:
        """Parse DeepSeek response"""
        choice = response.choices[0]
        message_text = choice.message.content if choice.message.content else ""
        function_call = None
        
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            function_call = {
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "call_id": getattr(call, "id", None)
            }
        
        result = {"message": message_text}
        if function_call:
            result["function_call"] = function_call
        
        return result or {"message": "No valid response returned from model."}