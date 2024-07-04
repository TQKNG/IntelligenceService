# Import packages for agent creation
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

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


class CreateSqlAgentService:
    def __init__(self):
        # Initialize the service, no attributes to set initially.
        return None
    
    def config_llm(self, api_key):
        # Configure the language model (LLM) with the provided API key and specific model settings.
        self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo-16k-0613", temperature=0)
    
    def config_db(self, connection_string):
        # Configure the database connection using the provided connection string.
        self.db = SQLDatabase.from_uri(connection_string)

    def config_system_prefix(self):
        # Set the system prefix used by the LLM to generate SQL queries.
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

        If the question does not seem related to the database, just return "I don't know" as the answer."""

    def get_table_names(self):
        # Retrieve the names of all usable tables in the configured database.
        return self.db.get_usable_table_names()
    
    def create_custom_retriever_tool(self, text):
        # Create a custom retriever tool to look up proper nouns using FAISS and OpenAI embeddings.
        self.vector_store = FAISS.from_texts(text, OpenAIEmbeddings())

        # Configure the retriever to return the top 5 most similar results.
        self.retriever = self.vector_store.as_retriever(
           search_kwargs={"k": 5} 
        )

        # Define the retriever tool's purpose and name.
        description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
        valid proper nouns. Use the noun most similar to the search."""

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
                "input": "Find the highest temperature for globaldws office.",
                "query": "SELECT MAX(temperature) AS 'Highest_Temperature' FROM [dbo].[tbl_data] DA "
                          "INNER JOIN dbo.tbl_floor F ON DA.roomID = F.id "
                          "INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id "
                          "INNER JOIN dbo.tbl_client C ON B.client_id = C.id "
                          "WHERE C.title LIKE '%GlobalDWS%';"
            },
            # Example 2
            {
                "input": "Retrieve the client and building that has data in June 2024",
                "query": "SELECT DISTINCT C.title AS 'Client Name', DISTINCT B.name AS 'Building Name' FROM [dbo].[tbl_data] DA "
                          "INNER JOIN dbo.tbl_floor F ON DA.roomID = F.id"
                          "INNER JOIN dbo.tbl_room R ON F.id = R.floors_id "
                          "INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id "
                          "INNER JOIN dbo.tbl_client C ON B.client_id = C.id "
                          "WHERE DA.enqueuedTime_Stamp >= '2024-06-01' AND DA.enqueuedTime_Stamp < '2024-07-01';"
            },
            # Example 3
            {
                "input": "Find the rooms in GlobalDWS office that have the temperature above 21 degrees celcius in the first week of February.",
                "query": "SELECT R.room AS 'Room Name', AVG(D.temperature) AS 'Average Temperature' FROM dbo.tbl_room R "
                         "INNER JOIN dbo.tbl_floor F ON R.floors_id = F.id "
                         "INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id "
                         "INNER JOIN dbo.tbl_client C ON B.client_id = C.id "
                         "INNER JOIN dbo.tbl_data D ON R.id = D.roomID "
                         "WHERE C.title LIKE '%GlobalDWS%' AND D.enqueuedTime_Stamp >= '2024-02-01' AND D.enqueuedTime_Stamp < '2024-02-08' "
                         "GROUP BY R.room HAVING AVG(D.temperature) > 21"
            },
            # Example 4
            {
                "input": "What is the temperature in June 1st, 2024",
                "query": "SELECT temperature FROM [dbo].[tbl_data] WHERE enqueuedTime_Stamp = '2024-06-01';"
            },
            # Example 5
            {
                "input": "Get all the rooms belongs to GlobalDWS",
                "query": "SELECT DA.room AS 'Room Name', F.floor AS 'Floor Name', B.name AS 'Building Name', C.title AS 'Client Name' "
                         "FROM [dbo].[tbl_room] DA INNER JOIN dbo.tbl_floor F ON DA.floors_id = F.id "
                         "INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id "
                         "INNER JOIN dbo.tbl_client C ON B.client_id = C.id "
                         "WHERE C.title LIKE '%GlobalDWS%';"
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
            suffix="If the question does not seem related to the database, just return 'I don't know. How can I assist you with question about your database' as the answer."
        )
        
    def create_full_prompt(self, question):
        # Create the full prompt template by combining the few-shot prompt with system and user messages.
        table_names = self.get_table_names()
        self.fullprompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(
                prompt=self.few_shot_prompt
            ), ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        # Invoke the prompt with the required input variables.
        self.fullprompt.invoke(
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
            prompt=self.fullprompt,
            extra_tools=[self.retriever_tool],
            verbose=True,
            agent_type="openai-tools"
        )
    
    def execute(self, question):
        # Execute the agent with the given question and return the result.
        return self.agent({"input": question})


class CreateDataAnalysisAgentService(CreateSqlAgentService):
    def __init__(self):
        return None
    
    def create_db_engine(self, connection_string):
        self.engine = create_engine(connection_string)
    
    def config_system_prefix(self):
        self.system_prefix="""You are a data analyst agent designed to interact with historical data. You will be using the historical data to make further analysis and prediction. DO NOT generate any pylot charts. JUST give me final analysis
    """
        
    def config_llm(self, api_key):
        self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4-turbo-2024-04-09", temperature=0)

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
         

    
   