from .researcher_agent import ResearcherAgent
from .supervisor_agent import SupervisorAgent

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, name,config):
        if agent_type == "Supervisor":
            return SupervisorAgent(name,config)
        elif agent_type == 'Researcher':
            return ResearcherAgent(name,config)
        else:
            raise ValueError("Unknown Agent Type")