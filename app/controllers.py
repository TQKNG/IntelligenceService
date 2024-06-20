from fastapi import APIRouter
from app.services.create_agent_service import CreateSqlAgentService
from typing import Dict, Any

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



### Routes ###
@router.post("/asksqlagent")
def ask_sql_agent(payload: Dict[Any,Any]):
    question = payload['question']

    if question != "":
        sql_agent = CreateSqlAgentService()
        sql_agent.config_llm(openai_api_key)
        sql_agent.config_db(f"mssql+pymssql://{os.getenv("DB_USER")}:{os.getenv("DB_PASS")}@{os.getenv("DB_SERVER")}/{os.getenv("DB_DATABASE")}")
        sql_agent.config_system_prefix()

        ### Turn data from each table to list of keywords ###
        rooms = query_as_list(sql_agent.db,"SELECT room FROM tbl_room")
        floors = query_as_list(sql_agent.db,"SELECT floor FROM tbl_floor")
        buildings = query_as_list(sql_agent.db,"SELECT name FROM tbl_building")
        clients = query_as_list(sql_agent.db,"SELECT title FROM tbl_client")

        sql_agent.create_custom_retriever_tool(rooms + floors + buildings + clients)

        sql_agent.create_example_selector()
        sql_agent.create_few_shot_prompt()
        sql_agent.create_full_prompt(question)
        sql_agent.create_agent()
        result = sql_agent.execute(question)

        print("testtt result",result)

        return {"message":"Success", "data":result}



