from fastapi import APIRouter
from app.models import Question, Answer
from app.services.db_service import DBService

# Import packages for agent creation
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Import  packages for few-shot learning
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Import prompts
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()


@router.post("/askAgent")
def ask_agent(question:Question):
    if question.text != "":
        # Call DB service
        connection_prop = {
            # "database_user": os.getenv("DB_USER"),
            # "database_password": os.getenv("DB_PASS"),
            # "database_server": os.getenv("DB_SERVER"),
            # "database_db": os.getenv("DB_DATABASE")

            "database_user":"khanhnguyentrq",
            "database_password": "Khanh92!",
            "database_server": "azureapi.database.windows.net",
            "database_db":"azureaiassistant"
        }


        db = SQLDatabase.from_uri(f"mssql+pymssql://khanhnguyentrq:Khanh92!@azureapi.database.windows.net/azureaiassistant")
        
        # Few-shot prompts
        # Example_Selector
        examples = [
    {
        "input": "Find the highest temperature for globaldws office.",
        "query": "SELECT MAX(temperature) AS 'Highest_Temperature' FROM [dbo].[tbl_data] DA "
                  "INNER JOIN dbo.tbl_floor F ON DA.roomID = F.id "
                  "INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id "
                  "INNER JOIN dbo.tbl_client C ON B.client_id = C.id "
                  "WHERE C.title = 'GlobalDWS';"
    },
    {
        "input":"What is the temperature in June 1st, 2024",
        "query":"SELECT temperature FROM [dbo].[tbl_data] WHERE enqueuedTime_Stamp = '2024-06-01';"
    }
]
 
                   
        embedding_function = OpenAIEmbeddings()
        
        
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embedding_function,
            FAISS,
            k=5,
            input_keys=["input"],
        )
        
        system_prefix = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.

        Here are some examples of user inputs and their corresponding SQL queries:"""
        
        few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
        )
        
        full_prompt = ChatPromptTemplate.from_messages(
            [
                  SystemMessagePromptTemplate(prompt=few_shot_prompt),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
            ]
            
        )
        
        prompt_val = full_prompt.invoke(
        {
            "input": question.text,
            "dialect": "SQL",
            "top_k": 5,
            "agent_scratchpad": []  # 
        }
    )

        print(prompt_val.to_string())
         
        # Instantiate LLM
        llm = ChatOpenAI(openai_api_key="sk-proj-kf9yknjQHXCyqASw4jfMT3BlbkFJd3zhvmKFUC414OEaHyZM", model="gpt-3.5-turbo-16k-0613", temperature=0)
        
        # Instantiate SQL agent
        agent_executor = create_sql_agent(
            llm = llm,
            db=db,
            prompt=full_prompt,
            verbose=True, # keeo output as it is or minimal
            agent_type="openai-tools"
        )
        
        # NL response
        result=agent_executor.invoke({"input":question.text})
        
            
        return {"message":"Success", "data":result}