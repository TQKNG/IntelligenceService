from fastapi import APIRouter
from app.models import Question, Answer
from app.services.db_service import DBService
from typing import Dict, Any

# Import packages for agent creation
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Import packages example selectors and semantic similarity, vector stores and embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Import Streamming services
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncIterable
from langchain.callbacks import AsyncIteratorCallbackHandler


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

# System os and dotenv
import os
from dotenv import load_dotenv

# Import abstract syntax grammar
import ast
import re

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

### Helper functions ###
# Flattening and filtering truthy values:
def query_as_list(db,query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Send streamming message
async def send_message(message: str)->AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()

    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback]
    )

    task = asyncio.create_task(
        model.agenerate(messages =message)
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception:{e}")
    finally:
        callback.done.set()

    await task




### Routes ###
@router.post("/askagent")
def ask_agent(payload: Dict[Any, Any]):

    # Get question from payload
    question = payload['question']
    
    if question != "":
        ### DB service ###
        # Create DB connection properties
        connection_prop = {
            "database_user": os.getenv("DB_USER"),
            "database_password": os.getenv("DB_PASS"),
            "database_server": os.getenv("DB_SERVER"),
            "database_db": os.getenv("DB_DATABASE")
        }

        # Create DB connection
        db = SQLDatabase.from_uri(f"mssql+pymssql://{connection_prop['database_user']}:{connection_prop['database_password']}@{connection_prop['database_server']}/{connection_prop['database_db']}")

        # Get table names
        table_names = db.get_usable_table_names()
        
        ### Turn data from each table to list of keywords ###
        rooms = query_as_list(db,"SELECT room FROM tbl_room")
        floors = query_as_list(db,"SELECT floor FROM tbl_floor")
        buildings = query_as_list(db,"SELECT name FROM tbl_building")
        clients = query_as_list(db,"SELECT title FROM tbl_client")
        # projects = query_as_list(db,"SELECT project FROM tbl_project")
        # devices = query_as_list(db,"SELECT device FROM tbl_devices")
        # deployments = query_as_list(db,"SELECT deployment FROM tbl_deployments")



        ### Create custom retriever tool ###
        # Embedding and vector database creation
        vector_db = FAISS.from_texts(rooms + floors + buildings + clients , OpenAIEmbeddings())

        # Get top 5 matches keywords from input against vector database 
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
        valid proper nouns. Use the noun most similar to the search."""

        # Create retriever tool to search for proper nouns
        retriever_tool = create_retriever_tool(
            retriever=retriever,
            description=description,
            name="search_proper_noun"
        )


        # Generate Few-shot prompts
        # Create Example_Selector
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
 
        # Instantiate OpenAIEmbeddings 
        embedding_function = OpenAIEmbeddings()

        # Using SemanticSimilarityExampleSelector to match the input question with the examples
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embedding_function,
            FAISS,
            k=5,
            input_keys=["input"],
        )

        # Create System Prefix/ instructions        
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
        
        # Create Few-shot prompt``
        few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k","table_names"],
        prefix=system_prefix,
        suffix="",
        )

        # Create Full prompt
        full_prompt = ChatPromptTemplate.from_messages(
            [
                  SystemMessagePromptTemplate(prompt=few_shot_prompt),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
            ]
            
        )
        
        # Invoke the prompt
        prompt_val = full_prompt.invoke(
        {
            "input": question,
            "dialect": "SQL",
            "top_k": 5,
            "table_names": table_names,
            "agent_scratchpad": []  # 
        }
    )

        print(prompt_val.to_string())
         
        # Instantiate LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k-0613", temperature=0)
        
        # Instantiate SQL agent_executor
        agent_executor = create_sql_agent(
            llm = llm,
            db=db,
            extra_tools=[retriever_tool],
            prompt=full_prompt,
            verbose=True, # keep output as it is or minimal
            agent_type="openai-tools"
        )
        
        # NL response
        result=agent_executor.invoke({"input":question})

        print("result",result)

        # print('test result',result["output"])

        # # Send streamming message
        # generator = send_message(result["output"])

        # print('test generator',generator)
        # return StreamingResponse(generator, media_type="text/event-stream")
        
            
        return {"message":"Success", "data":result}