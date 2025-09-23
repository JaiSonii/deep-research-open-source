"""
Module to init chat models from openrouter and use various opensource
and free models for deep research
"""

from langchain_openai import ChatOpenAI
from typing_extensions import Optional
import os



def init_chat_model(model: str, api_key: Optional[str], temperature: float=0, base_url: str = 'https://openrouter.ai/api/v1'):
    """Chat model initialization similar to langchain init_chat_model, but specific to openrouter"""
    open_router_key = api_key
    if not open_router_key and not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError('API Key not provided, either provide OPENROUTER_API_KEY as evironment variable or provide in the function')        
    return ChatOpenAI(
        model = model,
        temperature = temperature,
        api_key = open_router_key if open_router_key else os.getenv('OPENROUTER_API_KEY'),
        base_url= base_url
    )
