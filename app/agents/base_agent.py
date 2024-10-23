from dotenv import load_dotenv
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI

load_dotenv()

class BaseAgent(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        
        # Assuming `config` is a dictionary and keys should be accessed using `.get()`
        llm_config = config.get('llm', {})
        llm_provider = llm_config.get('provider')
        llm_model = llm_config.get('model')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 1000)
        
        # Handling different LLM providers
        if llm_provider == 'OpenAI':
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
        # Add other LLM service provider if needed
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
    @abstractmethod
    def perform_task(self, input_data):
        """Abstract method that must be implemented by subclasses"""
        pass
