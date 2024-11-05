from app.agents.agent_factory import AgentFactory
from app.chains.graph_builder import GraphBuilder

class MultiAgentService:
    def __init__ (self):
        self.agents = []
        self.workflow = None
        self.graph = None

    # Agent Creation and Initialization Logic
    def initialize_agents(self):
        agent = AgentFactory.create_agent(agent_type='Supervisor', name='Agent 1', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 0.7,
            'max_tokens': 1000
        })

        self.agents.append(agent)


    # Handle response
    def chatbot(self, state):
        return {"messages":[self.agents[0].invoke(state['messages'])]}

    # Create Workflow and Graph
    def construct_graph(self):
        self.graph = GraphBuilder()
        
        supervisor_node = self.agents[0].supervisor_agent
        
        self.graph.add_node('Supervisor',supervisor_node)

        self.graph.add_edge('test','tes2')

        self.graph.compile_graph()

    def draw_graph(self):
        self.graph.print_graph()


   

    # Invoke Response
    def invoke(self, question):
        supervisor_agent = self.initialize_agents()

        return supervisor_agent.invoke(question)