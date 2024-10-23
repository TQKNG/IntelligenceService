from typing_extensions import TypedDict
from typing import Annotated, Sequence

# Utilities
from utils.draw_img import draw_img

# Langchain libraries
from langchain.core.messages import HumanMessage, BaseMessage

# Langgraph libraries
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


class GraphBuilder:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def __init__(self):
        self.graph = None
        self.state = self.State(messages=[])
 
    
    def create_graph(self,State:State):
        self.graph = StateGraph(State)

    def create_node(self):
        pass

    def create_edge(self):
        pass

    def create_conditional_edge(self):
        pass

    def create_tool_node(self):
        pass

    def add_node(self):
        pass

    def add_edge(self):
        pass

    def add_conditional_edge(self):
        pass

    def compile_graph(self):
        pass

    def stream_graph(self):
        pass

    def print_graph(self):
        data = self.graph.get_graph()
        draw_img(data)