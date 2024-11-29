from app.agents.agent_factory import AgentFactory
from app.chains.graph_builder import GraphBuilder


class MultiAgentService:
    def __init__ (self):
        self.agents = []
        self.graph = None

    # Agent Creation and Initialization Logic
    def initialize_agents(self):
        agent_1 = AgentFactory.create_agent(agent_type='Supervisor', name='Supervisor', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 1,
            'max_tokens': 1000
        })

        agent_2 = AgentFactory.create_agent(
            agent_type='Researcher', name='Researcher', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 0.7,
            'max_tokens': 1000
        }
        )
        agent_3 = AgentFactory.create_agent(
            agent_type='SQLAgent', name='SQLAgent', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 1,            
        }
        )

        self.agents.append(agent_1)
        self.agents.append(agent_2)
        self.agents.append(agent_3)


    # Create Workflow and Graph
    def construct_graph(self):
        self.graph = GraphBuilder()

        # Create agent
        supervisor_agent = self.agents[0].create_agent()
        
        research_agent = self.agents[1].create_agent()

        sql_agent = self.agents[2].create_agent()

        # Create agent node
        supervisor_node = self.graph.create_tool_node(supervisor_agent,name='Supervisor')

        research_node = self.graph.create_tool_node(research_agent, name = 'Researcher')

        sql_node = self.graph.create_tool_node(sql_agent, name='SQLAgent')

        # Add node to graph
        self.graph.add_node('Supervisor',supervisor_node)

        self.graph.add_node('Researcher', research_node)

        self.graph.add_node('SQLAgent',sql_node)


        # Add Edges
        for member in self.agents:
            if member.agent_type == 'Supervisor':
                self.graph.add_edge('start','Supervisor')
            else:
                self.graph.add_edge(member.agent_type,'Supervisor')

        # Add condition edge
        conditional_edges = self.graph.create_conditional_edge(self.agents)
       

        self.graph.add_conditional_edge("Supervisor",lambda x: x['next'],conditional_edges)
        
        self.graph.compile_graph()

    def draw_graph(self):
        self.graph.print_graph()


    # Invoke Response
    def invoke(self, question):
        # Initialize Agents
        self.initialize_agents()

        self.construct_graph()

        # self.draw_graph()

        self.graph.stream_graph(question)
       
        return {'messages': 'Done'}