from .researcher_agent import ResearcherAgent
from .supervisor_agent import SupervisorAgent

class AgentFactory:
    @staticmethod
    def create_agent(agent_type):
        if agent_type == "Supervisor":
            return SupervisorAgent()
        elif agent_type == 'Researcher':
            return ResearcherAgent()
        else:
            raise ValueError("Unknown Agent Type")