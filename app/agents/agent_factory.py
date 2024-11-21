from .researcher_agent import ResearcherAgent
from .supervisor_agent import SupervisorAgent
from .api_agent import APIAgent

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, name,config):
        if agent_type == "Supervisor":
            return SupervisorAgent(name,agent_type,config)
        elif agent_type == 'Researcher':
            return ResearcherAgent(name,agent_type,config)
        elif agent_type == 'API':
            return APIAgent(name,agent_type,config)
        else:
            raise ValueError("Unknown Agent Type")