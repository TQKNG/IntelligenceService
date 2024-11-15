from app.agents.agent_factory import AgentFactory
from app.chains.graph_builder import GraphBuilder
import functools
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage

class MultiAgentService:
    def __init__ (self):
        self.agents = []
        self.workflow = None
        self.graph = None

    # Agent Creation and Initialization Logic
    def initialize_agents(self):
        agent_1 = AgentFactory.create_agent(agent_type='Supervisor', name='Agent 1', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 0.7,
            'max_tokens': 1000
        })

        agent_2 = AgentFactory.create_agent(
            agent_type='Researcher', name='Agent 2', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 0.7,
            'max_tokens': 1000
        }
        )
        agent_3 = AgentFactory.create_agent(
            agent_type='API', name='Agent 3', config={
            'llm': {'provider': 'OpenAI', 'model': 'gpt-4o-mini'},
            'temperature': 0.7,
            'max_tokens': 1000
        }
        )

        self.agents.append(agent_1)
        self.agents.append(agent_2)
        self.agents.append(agent_3)


    # Handle response
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        return {
            "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
        }

    # Create Workflow and Graph
    def construct_graph(self):
        self.graph = GraphBuilder()
        
        # supervisor_agent = self.agents[0].supervisor_agent()

        research_agent = self.agents[1].create_agent()

        api_agent = self.agents[2].create_agent()

        research_node = functools.partial(self.agent_node, agent=research_agent, name='Researcher')

        api_node = functools.partial(self.agent_node,agent=api_agent,name='API' )
        
        self.graph.add_node('Supervisor',self.agents[0].supervisor_agent)

        self.graph.add_node('Researcher', research_node)

        self.graph.add_node('API',api_node)


        # Add Edges
        self.graph.add_edge(START,'Supervisor')

        members = ["Researcher", 'API']
        for member in members:
            self.graph.add_edge(member,'Supervisor')

        # Add condition
        conditional_map ={k:k for k in members}
        conditional_map['FINISH'] = END

        self.graph.add_conditional_edge("Supervisor",lambda x: x['next'],conditional_map)
        
        self.graph.compile_graph()

    def draw_graph(self):
        self.graph.print_graph()


    # Invoke Response
    def invoke(self, question):
        # Initialize Agents
        self.initialize_agents()

        self.construct_graph()

        # self.draw_graph()

        # # Generate prompt-supervisor
        self.agents[0].generate_prompt()

        # # Define a chain
        self.agents[0].define_chain()

        state = {
            "messages": [
            {"role": "user", "content": question}
            ]
        }

        # # Run the chain
        # return self.agents[0].supervisor_agent(state)

        # return self.agents[0].invoke(question)
        self.graph.stream_graph(question)
       
        return {'messages': 'Done'}