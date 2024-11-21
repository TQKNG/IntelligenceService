from .base_agent import BaseAgent
from langgraph.prebuilt import create_react_agent
from app.tools.agent_tool import external_api_tool


class APIAgent(BaseAgent):
    def __init__(self, name, agent_type, config: dict):
        super().__init__(name, agent_type, config)
        
    def create_agent(self):
        return create_react_agent(self.llm,tools=[external_api_tool])

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass

    def invoke(self):
        pass