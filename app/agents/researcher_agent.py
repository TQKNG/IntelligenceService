from .base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass