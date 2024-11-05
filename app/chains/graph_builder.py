from typing_extensions import TypedDict
from typing import Annotated, Literal, Sequence
import operator
from pydantic import BaseModel
# Utilities
from app.utils.draw_img import draw_img

# Langchain libraries
from langchain_core.messages import BaseMessage ,HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langgraph libraries
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent


class GraphBuilder:
    @staticmethod
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next:str

    
    def __init__(self):
        self.workflow = StateGraph(GraphBuilder.AgentState)
        self.graph = None

    def create_edge(self):
        pass

    def create_conditional_edge(self):
        pass


    def create_tool_node(self):
        pass

    def add_node(self, name,node):
        self.workflow.add_node(name, node)

    def add_edge(self,start, end):
        self.workflow.add_edge(START,'Supervisor')
        self.workflow.add_edge('Supervisor', END)

    def add_conditional_edge(self):
        pass

    def compile_graph(self):
        self.graph = self.workflow.compile()

    def stream_graph(self):
        pass

    def print_graph(self):
        data = self.graph.get_graph()
        draw_img(data)