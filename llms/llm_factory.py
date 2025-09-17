from typing import Optional
from interfaces.i_llm import ILlm
from config import config

def create_llm(model_name: str) -> Optional[ILlm]:
    """Factory to create appropriate LLM instance"""
    
    model_config = config.get_llm_config(model_name)
    if not model_config:
        raise ValueError(f"Unknown model: {model_name}")
    
    provider = model_config.get('provider')
    
    # Import only what's needed
    if provider == 'openai':
        from llms.llm_openai import LlmOpenAi
        return LlmOpenAi(model_name)
    
    elif provider == 'anthropic':
        from llms.llm_anthropic import LlmAnthropic
        return LlmAnthropic(model_name)
    
    elif provider == 'google':
        from llms.llm_gemini import LlmGemini
        return LlmGemini(model_name)
    
    elif provider == 'deepseek':
        from llms.llm_deepseek import LlmDeepSeek
        return LlmDeepSeek(model_name)
    
    elif provider == 'xai':
        from llms.llm_grok import LlmGrok
        return LlmGrok(model_name)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")