from agents.researcher_agent import ResearcherAgent
from .base_chain import BaseChain

class ResearcherChain(BaseChain):
    def __init__(self):
        self.agent = ResearcherAgent()

    def run(self, input_data):
        pass
