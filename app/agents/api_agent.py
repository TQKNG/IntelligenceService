from .base_agent import BaseAgent
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from app.tools.agent_tool import external_api_tool


class APIAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)
        
    def create_agent(self):
        return create_react_agent(self.llm,tools=[external_api_tool])

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass

    def invoke(self):
        pass