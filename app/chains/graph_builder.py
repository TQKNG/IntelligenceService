from typing_extensions import TypedDict
from typing import Annotated, Literal, Sequence
from pydantic import BaseModel
# Utilities
from utils.draw_img import draw_img

# Langchain libraries
from langchain.core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langgraph libraries
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


class GraphBuilder:
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    class routeResponse(BaseModel):
        next: Literal["FINISH","Supervisor", "Researcher"]


    def __init__(self, members):
        self.graph = None
        self.state = self.State(messages=[])
        self.members = members
        self.options=None
        self.prompt= None
 
    
    def create_graph(self,State:State):
        self.graph = StateGraph(State)

    def create_node(self):
        pass


    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        return{
            "messages":[HumanMessage(content=result["messages"][-1].content,name=name)]
        }
    
    
    def define_options(self):
        self.options = ["FINISH"] + self.members

    def generate_prompt(self, system_prompt):
        self.prompt = ChatPromptTemplate.from_messages([
           ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of:{options}")
        ]).partial(options=str(self.options), members=", ".join(self.members))

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