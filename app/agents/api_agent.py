from .base_agent import BaseAgent
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import APIChain
from app.tools.agent_tool import tavily_tool
from app.docs.api_docs import my_api_docs

class APIAgent(BaseAgent):
    def __init__(self, name, config: dict):
        super().__init__(name, config)
        self.api_url_template="""
        Given the following API Documentation for my backend API: {my_api_docs}
        Your task is to construct the most efficient API URL to answer 
        the user's question, ensuring the 
        call is optimized to include only necessary information.
        Question: {question}
        API URL:
        """
        
    def create_agent(self):
        return create_react_agent(self.llm,tools=[tavily_tool])

    def generate_prompt(self):
        self.api_url_prompt = PromptTemplate(input_variables=['my_api_docs','question'], template=self.api_url_template)

    def define_chain(self):
        chain = APIChain.from_llm_and_api_docs(self.llm,my_api_docs)

    def invoke(self):
        pass