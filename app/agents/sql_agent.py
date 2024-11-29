import os
from dotenv import load_dotenv
from .base_agent import BaseAgent


load_dotenv()

class SQLAgent(BaseAgent):
    def __init__(self, name, agent_type, config: dict):
        super().__init__(name, agent_type, config)
        
    def create_agent(self):
        pass

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass

    def invoke(self):
        pass