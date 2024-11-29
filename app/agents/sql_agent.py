import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

load_dotenv()

class SQLAgent(BaseAgent):
    def __init__(self, name, agent_type, config: dict):
        super().__init__(name, agent_type, config)

        self.system_prefix  = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
      
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.

        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        """
        
    def create_agent(self):
        connection_string = (f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}?timeout=3")
        
        self.db = SQLDatabase.from_uri(connection_string,include_tables=['health_data_view'], view_support=True)

        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        return create_react_agent(self.llm, tools= toolkit.get_tools(),state_modifier=self.system_prefix)

    def generate_prompt(self):
        pass

    def define_chain(self):
        pass

    def invoke(self):
        pass