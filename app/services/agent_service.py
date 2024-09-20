# Import packages for agent creation
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import packages example selectors and semantic similarity, vector stores and embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Import retriever toolkits
from langchain.agents.agent_toolkits import create_retriever_tool


# Import prompts and templates
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# SQL Alchemy Lib
from sqlalchemy import create_engine
from langchain_experimental.agents import create_pandas_dataframe_agent

# Import Agent Type from langchain
from langchain.agents.agent_types import AgentType

# Utils Lib
from typing_extensions import TypedDict
from typing import Literal
import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_state: str

def node_1(state):
    print("node_1")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("node_2")
    return {"graph_state": state['graph_state'] +" happy"}

def node_3(state):
    print("node_3")
    return {"graph_state": state['graph_state'] +" sad"}

def decide_mood(state)-> Literal["node_2","node_3"]:
    user_input = state['graph_state']

    if random.random() < 0.5:
        return "node_2"
    
    return "node_3"


agent_type =["Supervisor","General","SQL","Data"]
system_prompt =(
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options =["FINISH"] + agent_type

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

# Supervisor node -> General/SQL/Data node


class BaseAgent:
    def __init__(self, id:str, name:str, open_api_key: str,model:str, temperature:float, max_tokens=350 ):
        self._id = id
        self._name = name
        self._status= 'active'
        self._open_api_key = open_api_key
        self._model=model
        self._temperature=temperature
        self._max_tokens = max_tokens
        self._system_prefix=None
        self._llm = None
        self._streaming=True

    @property
    def id(self)->str:
        return self._id
    
    @property
    def name(self)->str:
        return self._name
    
    @property
    def status(self)->str:
        return self._status
    
    @status.setter
    def status(self, value:str):
        self._status = value

    def initialize(self):
        """Setup method for initializing agent"""
        pass

    def perform_action(self, action:str, *args):
        """Perform actions"""
        pass

    def cleanup(self):
        """Cleanup method for releasing resources"""
        pass


class GeneralContextAgent(BaseAgent):
    def __init__(self, id:str, name:str, open_api_key:str,model:str, temperature:float, max_token:int):
        super().__init__(id, name, open_api_key, model, temperature, max_token)
        self.full_prompt = None
        self._builder = None
        self._graph = None

    def initialize(self):
        self._llm = ChatOpenAI(openai_api_key=self._open_api_key, model=self._model, temperature=self._temperature, max_tokens = self._max_tokens, streaming = self._streaming)

        self._system_prefix = """You are an agent designed to answer general questions.
        Provide shortest answer as possible
        """
        
        # Build graph
        self._builder = StateGraph(State)
        self._builder.add_node("node_1", node_1)
        self._builder.add_node("node_2", node_2)
        self._builder.add_node("node_3", node_3)

        # Logic
        self._builder.add_edge(START,node_1)
        self._builder.add_conditional_edges("node_1", decide_mood)
        self._builder.add_edge("node_2", END)
        self._builder.add_edge("node_3", END)

        
    def perform_action(self, action:str, *args):
        match action:
            case "query":
                question = args[0].get('question') if len(args) > 0 else None
                
                print("test question", question)
                self._messages = [
                SystemMessage(content=self._system_prefix),
                HumanMessage(content=f"{question}"),
                ]
                response= self._llm.invoke(self._messages)
                print("test response", response)
                
                pass
            case _:
                # Add
                print(self._builder)
                self._graph = self._builder.compile()

            # View
                display(Image(self._graph.get_graph().draw_mermaid_png()))

    def cleanup(self):
        print("General Context Cleanup")


class SQLAgent(BaseAgent):
    def __init__(self, id:str, name:str, db_connection):
        super().__init__(id, name)
        self.db_connection = db_connection

    def initialize(self):
        print("SQL Agent Initialize")

    def perform_action(self, action:str, *args):
        print("SQL Agent Action")

    def connect_to_db(self):
        pass

    def execute_query(self, query:str):
        pass

    def cleanup(self):
        print("SQL Agent Cleanup")
        

class CreateSqlAgentService:
    def __init__(self):
        # Initialize the service, no attributes to set initially.
        self.llm = None
        self.db = None
        self.retriever = None
        self.full_prompt = None
        self.agent = None
        self.clients = None
    
    def config_llm(self, api_key, model):
        # Configure the language model (LLM) with the provided API key and specific model settings.
        # self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4-turbo-2024-04-09", temperature=0, max_retries=2)
         # self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
        # self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4-turbo-2024-04-09", temperature=0, streaming = True)

        self.llm = ChatOpenAI(openai_api_key=api_key, model=model, temperature=0, streaming = True)
   
    
    def config_db(self, connection_string):
        # Configure the database connection using the provided connection string.
        self.db = SQLDatabase.from_uri(connection_string,include_tables=['health_data_view'], view_support=True)

    def config_system_prefix(self):
        # Set the system prefix used by the LLM to generate SQL queries.
        # self.system_prefix = """You are an agent designed to interact with a SQL database with ability to analyze data to provide insights.
        # Given an input question, 
        # Step 1: You need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! You have access to the following tables: {table_names}
        # Step 2: Before generate any query, refer the example queries to learn how to generate the query.
        # Step 3: You need to create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        # Step 4: You can order the results by a relevant column to return the most interesting examples in the database.
        
        # Note: 
        # - Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        # - You can aggregate the results to get the answer.
        # - You can run multiple query when needed to compare data.
        # - You have access to tools for interacting with the database.
        # - Only use the given tools. Only use the information returned by the tools to construct your final answer.
        # - You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        # - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        # If the question does not seem related to the database, just return "I don't know" as the answer."""


        ###Prompt 2
        self.system_prefix = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

        You have access to the following tables: {table_names}

        If the question requires more analysis, you have the ability to analyze the data to provide insights.
        """


    def get_table_names(self):
        # Retrieve the names of all usable tables in the configured database.
        return self.db.get_usable_table_names()
    
    
    def set_client_names(self,clients):
        self.clients=clients
    
    def create_custom_retriever_tool(self, text):
        # Create a custom retriever tool to look up proper nouns using FAISS and OpenAI embeddings.
        self.vector_store = FAISS.from_texts(text, OpenAIEmbeddings())

        # Configure the retriever to return the top 5 most similar results.
        self.retriever = self.vector_store.as_retriever(
           search_kwargs={"k": 5} 
        )

        # Define the retriever tool's purpose and name.
        description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search.
        Note:
        - Check all the nouns in the question text
        """

        self.retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            description=description,
            name="search_proper_nouns",
        )

    def create_example_selector(self):
        # Create an example selector using semantic similarity for few-shot learning.
        embedding_function = OpenAIEmbeddings()
        examples = [
            # Example 1
            {
                "input": "How many rooms in contoso are with average temperature higher than 21 degree",
                "query": "SELECT COUNT(DISTINCT room_name) AS NumberOfRooms FROM health_data_view WHERE client_name = 'Contoso' AND temperature > 21 GROUP BY room_name HAVING AVG(temperature) > 21"
            },
            # Example 2
            {
                "input": "What is the average, highest, and lowest temperature in globaldws in April 2024",
                "query": "SELECT AVG(temperature) AS 'Average Temperature', MAX(temperature) AS 'Highest Temperature', MIN(temperature) AS 'Lowest Temperature' FROM health_data_view WHERE client_name = 'GlobalDWS' AND updated_time BETWEEN '2024-04-01' AND '2024-04-30'"
            },
            # Example 3
            {
                "input": "Provided that if the average temperature lower than 27 in the last 3 days, an email notification will be triggered. How many time email should be triggered in March 24 for GlobalDWS",
                "query": "SELECT COUNT(*) AS EmailTriggers FROM (SELECT AVG(temperature) AS AvgTemperature, CAST(updated_time AS DATE) AS DateOnly FROM health_data_view WHERE client_name = 'GlobalDWS' AND updated_time BETWEEN '2024-03-01' AND '2024-03-31' GROUP BY CAST(updated_time AS DATE) HAVING AVG(temperature) < 27) AS SubQuery"
            },
            # Example 4
            {
                "input": "In 2024, between GlobalDWS and Contoso which had higher co2 level",
                "query": "SELECT AVG(co2) AS Average_CO2 FROM health_data_view WHERE client_name = 'GlobalDWS' AND updated_time BETWEEN '2024-01-01' AND '2024-12-31'; SELECT AVG(co2) AS Average_CO2 FROM health_data_view WHERE client_name = 'Contoso' AND updated_time BETWEEN '2024-01-01' AND '2024-12-31';"
            },
            # Example 5
            {
                "input": "What is the trend of temperature in globaldws in April 2024",
                "query": "SELECT CAST(updated_time AS DATE) AS Date, AVG(temperature) AS AverageTemperature FROM health_data_view WHERE client_name = 'GlobalDWS' AND updated_time BETWEEN '2024-04-01' AND '2024-04-30' GROUP BY CAST(updated_time AS DATE) ORDER BY Date"
            },
            {
                "input": "Which day of the week the temperature is highest for PSPC",
                "query": "SELECT DATENAME(dw, updated_time) AS DayOfWeek, MAX(temperature) AS MaxTemperature FROM health_data_view WHERE client_name = 'PSPC - Public Services and Procurement Canada' AND updated_time BETWEEN '2024-05-01' AND '2024-05-31' GROUP BY DATENAME(dw, updated_time) ORDER BY MaxTemperature DESC"
            }
        ]
        # Use the examples to create a semantic similarity example selector.
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embedding_function,
            FAISS,
            k=5,
            input_keys=["input"],
        )
    
    def create_few_shot_prompt(self):
        # Create a few-shot learning prompt template using the example selector.
        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k", "table_names"],
            prefix=self.system_prefix,
            suffix="If the question does not seem related to the database, retry"
        )
        
    def create_full_prompt(self, question):
        # Create the full prompt template by combining the few-shot prompt with system and user messages.
        table_names = self.get_table_names()
        self.full_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(
                prompt=self.few_shot_prompt
            ), ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        # Invoke the prompt with the required input variables.
        self.full_prompt.invoke(
        {
            "input": question,
            "dialect": "SQL",
            "top_k": 5,
            "table_names": table_names,
            "agent_scratchpad": []  # Placeholder for the agent's scratchpad messages.
        })

    def create_agent(self):
        # Create the SQL agent using the configured LLM, database, prompt, and additional tools.
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            prompt=self.full_prompt,
            extra_tools=[self.retriever_tool],
            verbose=True,
            agent_type="openai-tools"
        )
    
    def execute(self, question):
        # Execute the agent with the given question and return the result.
        return self.agent({"input": question})


class CreateDataAnalysisAgentService(CreateSqlAgentService):
    def __init__(self):
        self.agent = None
    
    def create_db_engine(self, connection_string):
        self.engine = create_engine(connection_string)
    
    def config_system_prefix(self):
        self.system_prefix="""You are a data analyst agent designed to interact with historical data. You will be using the historical data to make further analysis and prediction. DO NOT generate any pylot charts. JUST give me final analysis
    """
        
    def config_llm(self, api_key):
        self.llm = ChatOpenAI(openai_api_key=api_key, model="o1-mini", temperature=0.5, streaming=True)

    def create_agent(self,df):
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,
            prefix=self.system_prefix,
            number_of_head_rows=300
        )

    def execute(self, question):
        return self.agent.invoke(question)
         

    
   