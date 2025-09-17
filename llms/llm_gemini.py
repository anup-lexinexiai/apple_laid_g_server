
from google import genai
from llms.base_llm import BaseLLM
from typing import Dict, Any
import base64
import logging
import json
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateContentConfig, Tool, FunctionDeclaration, SafetySetting

logger = logging.getLogger(__name__)

class LlmGemini(BaseLLM):
    """
    Google Gemini models using the modern google-genai SDK.
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        if not self.api_key:
            logger.error(f"API key for Gemini model '{self.model_name}' not found.")
            raise ValueError(f"API key for Gemini model '{self.model_name}' is required.")
        
        # The new SDK uses a client-based approach
        # Note: The library uses genai.configure(api_key=...) behind the scenes
        # if you use top-level functions, but a client is cleaner.
        # Let's stick to the direct model initialization which is also common.
        genai.configure(api_key=self.api_key)

        self.instruction = None
        logger.info(f"LlmGemini initialized for model '{self.model}'. Max output tokens: {self.max_tokens}")

    def llm_response(self, complete_chat: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conversation_history = complete_chat.get("conversation_history", [])
            instruction = complete_chat.get("instruction")
            img_data = complete_chat.get('img_data')
            available_functions = complete_chat.get("available_functions", [])

            # --- Function Declarations (New SDK Format) ---
            tools = []
            if available_functions:
                function_declarations = [
                    self._format_function_declaration(func) for func in available_functions if func.get("name")
                ]
                if function_declarations:
                    tools.append(Tool(function_declarations=function_declarations))

            if instruction:
                self.instruction = instruction

            # --- Safety Settings ---
            safety_settings = [
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE)
            ]

            # --- Generation Config ---
            generation_config = GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                # response_mime_type="text/plain", # Let the model decide based on content
            )

            # Initialize the model with all configurations
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self.instruction,
                tools=tools if tools else None
            )

            # --- History and Content Processing ---
            # The new SDK can take the whole history at once.
            contents = self._process_history_for_sdk(conversation_history)
            
            # Handle images in the last message if they exist
            if img_data and contents:
                 self._add_images_to_last_content(img_data, contents)

            logger.debug(f"Making generate_content call to '{self.model}'")
            response = model.generate_content(contents=contents)
            logger.debug(f"Raw response from SDK: {response}")

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Critical Error in LlmGemini llm_response: {e}", exc_info=True)
            return {"message": f"Gemini Critical Error: {str(e)}", "function_call": None}

    def _format_function_declaration(self, func: Dict) -> FunctionDeclaration:
        """Formats a single function dictionary into a FunctionDeclaration object."""
        props = {}
        required = []
        for param_name, param_info in func.get("parameters", {}).items() or {}.items():
            if isinstance(param_info, dict):
                props[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", "")
                }
                if param_info.get("required", False):
                    required.append(param_name)
        
        return FunctionDeclaration(
            name=func["name"],
            description=func.get("description", ""),
            parameters={"type": "object", "properties": props, "required": required}
        )

    def _process_history_for_sdk(self, conversation_history: list) -> list:
        """Converts the internal conversation history format to the Gemini SDK format."""
        contents = []
        for msg in conversation_history:
            role = msg.get("role")
            if role == "assistant":
                role = "model"
            
            if role not in ["user", "model", "function"]:
                logger.warning(f"Skipping history message with invalid role: {role}")
                continue

            content_parts = []
            msg_type = msg.get("type")

            if msg_type == "function_call":
                role = "model" # A function call is always from the model
                fc_name = msg.get("name")
                fc_args = msg.get("arguments", {})
                if isinstance(fc_args, str):
                    try: fc_args = json.loads(fc_args)
                    except json.JSONDecodeError: fc_args = {}
                
                if fc_name:
                    content_parts.append({"function_call": {"name": fc_name, "args": fc_args}})

            elif msg_type == "function_call_output":
                role = "function" # A function result has the 'function' role
                fco_name = msg.get("name")
                fco_output = msg.get("output", "")
                if fco_name:
                    content_parts.append({"function_response": {"name": fco_name, "response": {"content": fco_output}}})

            else: # Regular text message
                text_content = msg.get("content", "")
                if text_content:
                    content_parts.append({"text": text_content})

            if content_parts:
                contents.append({"role": role, "parts": content_parts})
        
        return contents

    def _add_images_to_last_content(self, img_data: Any, contents: list):
        """Adds image data to the last 'user' message in the contents list."""
        if not contents or contents[-1].get("role") != "user":
            # If there's no history or the last message isn't from the user, create a new one.
            contents.append({"role": "user", "parts": []})

        if not isinstance(img_data, list):
            img_data = [img_data]
        
        for b64_string in img_data:
            try:
                if not b64_string or not isinstance(b64_string, str): continue
                image_bytes = base64.b64decode(b64_string)
                contents[-1]["parts"].append({
                    "inline_data": {"mime_type": "image/png", "data": image_bytes}
                })
            except Exception as img_err:
                logger.error(f"Failed to process a base64 image. Error: {img_err}", exc_info=False)

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parses the response from the Gemini SDK into the standardized dictionary format."""
        result = {}
        message_text = ""
        function_call = None

        try:
            if not hasattr(response, 'candidates') or not response.candidates:
                logger.error("Response has no candidates.")
                return {"message": "Gemini Error: Model response has no candidates.", "function_call": None}

            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        call = part.function_call
                        args_dict = dict(call.args) if hasattr(call, 'args') else {}
                        function_call = {"name": call.name, "arguments": args_dict}
                        logger.info(f"Received function call request: {function_call}")
                    elif hasattr(part, 'text') and part.text:
                        message_text += part.text
            
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason and str(finish_reason) not in ["STOP", "1"]:
                 logger.warning(f"Response finished with non-standard reason: {finish_reason}")
                 if not message_text and not function_call:
                     message_text = f"Model stopped. Reason: {finish_reason}."

        except Exception as parse_err:
            logger.error(f"Error parsing SDK response: {parse_err}", exc_info=True)
            return {"message": f"Error parsing Gemini response: {parse_err}", "function_call": None}

        if message_text.strip():
            result["message"] = message_text.strip()
        if function_call:
            result["function_call"] = function_call
        
        if not result:
            logger.warning("No text message or function call generated by the model.")
            result["message"] = "" # Return empty message if nothing was generated

        return result
