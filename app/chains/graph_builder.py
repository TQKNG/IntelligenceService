from typing_extensions import TypedDict
from typing import Annotated, Sequence
import operator
import functools
# Utilities
from app.utils.draw_img import draw_img

# Langchain libraries
from langchain_core.messages import BaseMessage ,HumanMessage

# Langgraph libraries
from langgraph.graph import START, StateGraph, END

class GraphBuilder:
    @staticmethod
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next:str

    
    def __init__(self):
        self.workflow = StateGraph(GraphBuilder.AgentState)
        self.graph = None


    # Handle response
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        
        # Supervisor with return the routeResponse
        if name == 'Supervisor':
            return result
        
        # Other agent with return the response message
        return {
            "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
        }

    def create_tool_node(self,agent, name):
        return functools.partial(self.agent_node, agent=agent, name=name)

    def add_node(self, name,node):
        self.workflow.add_node(name, node)

    def add_edge(self,start, end):
        if start == 'start':
            self.workflow.add_edge(START,end)
        elif end == 'end':
            self.workflow.add_edge(start, END)
        else:
            self.workflow.add_edge(start, end)

    def create_conditional_edge(self,agents):
        conditional_edges ={agent.name: agent.name for agent in agents if agent.name != "Supervisor"}
        conditional_edges['FINISH'] = END
        return conditional_edges

    def add_conditional_edge(self, start, func, conditional_map):
        self.workflow.add_conditional_edges(start,func,conditional_map)

    def compile_graph(self):
        self.graph = self.workflow.compile()

    def stream_graph(self, question):
         for s in self.graph.stream(
            {
                "messages":[
                    HumanMessage(content=question)
                ]
            }
        ):
            if "__end__" not in s:
                print(s)
                print("----")


    def print_graph(self):
        data = self.graph.get_graph()
        draw_img(data)