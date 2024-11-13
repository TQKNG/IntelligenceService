from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from app.prompts import supervisor_prompt
from pydantic import BaseModel
from typing import Literal

members = ["Researcher", 'API']
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal["Researcher",'API']

class SupervisorAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)
        self.system_prompt=   """
        You are a supervisor tasked managing a conversation between following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH
         """
        self.prompt = None
        self.chain = None

    def create_agent(self):
       pass

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

    def supervisor_agent(self,state):
        self.generate_prompt()
        self.define_chain()
        return self.chain.invoke(state)
    
    def invoke(self, input):
        return self.llm.invoke(input)
        