from openai import OpenAI
from llms.base_llm import BaseLLM
from typing import Dict, Any, List, Optional
import json
import base64

class LlmOpenAi(BaseLLM):
    """OpenAI GPT models including O1, O3 reasoning models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=self.api_key)
        
        # Set role based on model type
        self.system_role = "developer" if self.reasoning_model else "system"
    
    def format_function_for_llm(self, func: Dict) -> Optional[Dict]:
        """Format function for OpenAI tools API"""
        function_name = func.get("name")
        function_description = func.get("description")
        parameters = func.get("parameters", {})
        
        if not function_name or not function_description:
            return None
        
        tool = {
            "type": "function",
            "name": function_name,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True  # For structured outputs
        }
        
        for param_name, param_info in parameters.items():
            tool["parameters"]["properties"][param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", "")
            }
            if param_info.get("required", False):
                tool["parameters"]["required"].append(param_name)
        
        return tool
    
    def process_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Process messages for OpenAI format"""
        processed = []
        
        for msg in messages:
            msg_type = msg.get("type")
            role = msg.get("role")
            
            if msg_type == "function_call":
                # Assistant made a function call
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
                # Function returned output
                processed.append({
                    "role": "tool",
                    "content": msg.get("output", ""),
                    "tool_call_id": msg.get("call_id")
                })
            elif role in ["user", "assistant"]:
                processed.append({
                    "role": role,
                    "content": msg.get("content")
                })
        
        return processed
    
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Main response method with full OpenAI capabilities"""
        try:
            conversation_history = complete_chat.get('conversation_history', [])
            instruction = complete_chat.get('instruction')
            img_data = complete_chat.get('img_data')  # List of base64 images
            available_functions = complete_chat.get('available_functions', [])
            
            # Process functions into tools
            tools = []
            if available_functions:
                for func in available_functions:
                    tool = self.format_function_for_llm(func)
                    if tool:
                        tools.append(tool)
            
            # Add web search if enabled
            if self.web_search:
                tools.append({
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium"
                })
            
            # Build messages
            messages = []
            
            # Add instruction as system message
            if instruction:
                messages.append({
                    "role": self.system_role,
                    "content": instruction
                })
            
            # Process conversation history
            messages.extend(self.process_conversation_history(conversation_history))
            
            # Handle images if present
            if img_data:
                image_content = []
                for b64_img in img_data:
                    image_content.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64_img}"
                    })
                
                messages.append({
                    "role": "user",
                    "content": image_content
                })
            
            # Make API call based on model type
            if self.priority:
                # Priority tier for faster responses
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    text={"format": {"type": "text"}},
                    reasoning={},
                    tools=tools if tools else None,
                    service_tier="priority",
                    store=False
                )
            elif self.reasoning_model:
                # O1, O3 reasoning models
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    text={"format": {"type": "text"}},
                    reasoning={"effort": "medium"},
                    tools=tools if tools else None,
                    store=False
                )
            else:
                # Standard chat completion
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
                "message": f"OpenAI Error: {str(e)}",
                "function_call": None
            }
    
    def parse_response(self, response) -> Dict[str, Any]:
        """Parse OpenAI response format"""
        message_text = None
        function_call = None
        
        # Handle new response format (O1, O3)
        if hasattr(response, 'output'):
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    message_text = item.content[0].text if item.content else ""
                elif getattr(item, "type", None) == "function_call":
                    function_call = {
                        "name": item.name,
                        "arguments": json.loads(item.arguments),
                        "call_id": getattr(item, "call_id", None)
                    }
        # Handle standard format
        elif hasattr(response, 'choices'):
            choice = response.choices[0]
            message = choice.message
            
            if message.content:
                message_text = message.content
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_call = {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                    "call_id": tool_call.id
                }
        
        result = {}
        if message_text:
            result["message"] = message_text
        if function_call:
            result["function_call"] = function_call
        
        if not result:
            result["message"] = "No response generated"
        
        return result