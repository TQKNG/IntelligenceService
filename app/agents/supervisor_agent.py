from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal

members = ["Researcher", 'SQLAgent']
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal["Researcher",'SQLAgent',"FINISH"]

class SupervisorAgent(BaseAgent):
    def __init__(self, name,agent_type, config: dict):
        super().__init__(name,agent_type, config)
        self.system_prompt=   """
        You are a supervisor tasked managing a conversation between following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When you receive the response from {members}, respond immediatly with FINISH
         """
        self.prompt = None
        self.chain = None

    def create_agent(self):
       self.generate_prompt()
       self.define_chain()
       return self.chain
       

    def generate_prompt(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",self.system_prompt,
            ),
            MessagesPlaceholder(variable_name='messages'),
            ("system",
             "Given the conversation above, who should act next?"
             "Or should we FINISH? Select one of: {options}")
        ]).partial(options=str(options), members=', '.join(members))

    def define_chain(self):
        self.chain = self.prompt | self.llm.with_structured_output(routeResponse)
    
    def invoke(self):
        pass
        