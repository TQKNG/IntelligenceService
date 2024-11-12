from .base_agent import BaseAgent
from langgraph.prebuilt import create_react_agent
from app.tools.agent_tool import tavily_tool

class ResearcherAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)

    def create_agent(self):
        return create_react_agent(self.llm,tools=[tavily_tool])

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass

    def invoke(self):
        pass