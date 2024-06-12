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

# Import retriever toolkits
from langchain.agents.agent_toolkits import create_retriever_tool

# Import prompts
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# System os and dotenv
import os
from dotenv import load_dotenv

# Import abstract syntax grammar
import ast
import re


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()


# Flattening and filtering truthy values:
def query_as_list(db,query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Format table names to string


@router.post("/askAgent")
def ask_agent(question:Question):
    if question.text != "":
        # Call DB service
        connection_prop = {
            "database_user": os.getenv("DB_USER"),
            "database_password": os.getenv("DB_PASS"),
            "database_server": os.getenv("DB_SERVER"),
            "database_db": os.getenv("DB_DATABASE")
        }


        db = SQLDatabase.from_uri(f"mssql+pymssql://{connection_prop['database_user']}:{connection_prop['database_password']}@{connection_prop['database_server']}/{connection_prop['database_db']}")

        # Get table names
        table_names = db.get_usable_table_names()
        
        # Training the model with examples
        rooms = query_as_list(db,"SELECT room FROM tbl_room")
        floors = query_as_list(db,"SELECT floor FROM tbl_floor")
        buildings = query_as_list(db,"SELECT name FROM tbl_building")
        clients = query_as_list(db,"SELECT title FROM tbl_client")


        # Create custom retriever tool
        vector_db = FAISS.from_texts(rooms + floors + buildings + clients, OpenAIEmbeddings())
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
        valid proper nouns. Use the noun most similar to the search."""

        retriever_tool = create_retriever_tool(
            retriever=retriever,
            description=description,
            name="search_proper_noun"
        )


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
    },
    {
        "input":"Get all the rooms belongs to GlobalDWS",
        "query":"SELECT DA.room AS 'Room Name', F.floor AS 'Floor Name', B.name AS 'Building Name', C.title AS 'Client Name' FROM [dbo].[tbl_room] DA INNER JOIN dbo.tbl_floor F ON DA.floors_id = F.id INNER JOIN dbo.tbl_building B ON F.buildings_id = B.id INNER JOIN dbo.tbl_client C ON B.client_id = C.id WHERE C.title = 'GlobalDWS';"
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

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

You have access to the following tables: {table_names}

If the question does not seem related to the database, just return "I don't know" as the answer."""
        
        few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k","table_names"],
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
            "table_names": table_names,
            "agent_scratchpad": []  # 
        }
    )

        print(prompt_val.to_string())
         
        # Instantiate LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k-0613", temperature=0)
        
        # Instantiate SQL agent
        agent_executor = create_sql_agent(
            llm = llm,
            db=db,
            extra_tools=[retriever_tool],
            prompt=full_prompt,
            verbose=True, # keep output as it is or minimal
            agent_type="openai-tools"
        )
        
        # NL response
        result=agent_executor.invoke({"input":question.text})
        
            
        return {"message":"Success", "data":result}