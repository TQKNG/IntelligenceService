from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate

class SupervisorAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)
    
    def perform_task(self, input_data):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a helpful assistant. Please give your answer in one sentence"),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "input": input_data
        })

        return response