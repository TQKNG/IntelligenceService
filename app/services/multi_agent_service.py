from app.agents.agent_factory import AgentFactory


class MultiAgentService:
    def __init__ (self):
        self.agents = []
        self.workflow = None

    # Agent Creation and Initialization Logic
    def initialize_agents(self):
        return AgentFactory.create_agent(agent_type='Supervisor', name='Agent 1', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-3.5-turbo-0301'},
            'temperature': 0.7,
            'max_tokens': 1000
        })


    # Handling Prompts and Query Processing


    # Streaming Response


    # Invoke Response
    def invoke(self, question):
        supervisor_agent = self.initialize_agents()

        supervisor_agent.perform_task(question)