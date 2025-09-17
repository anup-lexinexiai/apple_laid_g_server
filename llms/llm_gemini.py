import google.generativeai as genai
from llms.base_llm import BaseLLM
from typing import Dict, Any, List
import json
import base64
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class LlmGemini(BaseLLM):
    """Google Gemini models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=self.api_key)
        
        self.generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "response_mime_type": "text/plain"
        }
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        self.chat_session = None
    
    def format_function_for_llm(self, func: Dict) -> Dict:
        """Format function for Gemini"""
        props = {}
        required = []
        
        for param_name, param_info in func.get("parameters", {}).items():
            if isinstance(param_info, dict):
                props[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", "")
                }
                if param_info.get("required", False):
                    required.append(param_name)
        
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        }
    
    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        """Main response method for Gemini"""
        try:
            conversation_history = complete_chat.get('conversation_history', [])
            instruction = complete_chat.get('instruction')
            img_data = complete_chat.get('img_data')
            available_functions = complete_chat.get('available_functions', [])
            
            # Process functions
            function_declarations = []
            if available_functions:
                for func in available_functions:
                    if func.get("name"):
                        function_declarations.append(
                            self.format_function_for_llm(func)
                        )
            
            # Initialize model with tools
            tools = None
            if function_declarations:
                tools = [{"function_declarations": function_declarations}]
            
            self.model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config,
                tools=tools,
                safety_settings=self.safety_settings,
                system_instruction=instruction
            )
            
            # Process history
            history = self.process_gemini_history(conversation_history[:-1] if conversation_history else [])
            
            # Start chat
            self.chat_session = self.model.start_chat(history=history)
            
            # Process last message
            if conversation_history:
                last_msg = conversation_history[-1]
                
                if last_msg.get("type") == "function_call_output":
                    # Send function response
                    response = self.send_function_response(last_msg)
                else:
                    # Send regular message
                    content = last_msg.get("content", "")
                    
                    # Handle images
                    if img_data:
                        parts = [content]
                        for b64_img in img_data:
                            image_bytes = base64.b64decode(b64_img)
                            parts.append({"mime_type": "image/png", "data": image_bytes})
                        response = self.chat_session.send_message(parts)
                    else:
                        response = self.chat_session.send_message(content)
            else:
                # No history, send empty to start
                response = self.chat_session.send_message("")
            
            return self.parse_gemini_response(response)
            
        except Exception as e:
            return {
                "message": f"Gemini Error: {str(e)}",
                "function_call": None
            }
    
    def process_gemini_history(self, messages: List[Dict]) -> List:
        """Convert messages to Gemini format"""
        history = []
        
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                role = "model"
            
            msg_type = msg.get("type")
            
            if msg_type == "function_call":
                # Function call from model
                history.append({
                    "role": "model",
                    "parts": [{
                        "function_call": {
                            "name": msg.get("name"),
                            "args": json.loads(msg.get("arguments", "{}"))
                        }
                    }]
                })
            elif msg_type == "function_call_output":
                # Function response
                history.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": msg.get("name"),
                            "response": {"content": msg.get("output", "")}
                        }
                    }]
                })
            else:
                # Regular message
                history.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
        
        return history
    
    def send_function_response(self, msg: Dict):
        """Send function response to Gemini"""
        from google.generativeai import protos
        from google.protobuf import struct_pb2
        
        function_name = msg.get("name")
        output_content = msg.get("output", "")
        
        response_struct = struct_pb2.Struct()
        response_data = {"content": output_content}
        response_struct.update(response_data)
        
        function_response_proto = protos.FunctionResponse(
            name=function_name,
            response=response_struct
        )
        
        function_response_part = protos.Part(
            function_response=function_response_proto
        )
        
        return self.chat_session.send_message(function_response_part)
    
    def parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse Gemini response"""
        result = {}
        message_text = ""
        function_call = None
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        call = part.function_call
                        args_dict = dict(call.args) if hasattr(call, 'args') else {}
                        function_call = {
                            "name": call.name,
                            "arguments": args_dict,
                            "call_id": ""
                        }
                    elif hasattr(part, 'text'):
                        message_text += part.text
        
        if message_text:
            result["message"] = message_text
        if function_call:
            result["function_call"] = function_call
        
        return result