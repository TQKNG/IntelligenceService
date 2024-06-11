from fastapi import APIRouter
from app.models import Question, Answer
from app.services.db_service import DBService
# from langchain.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

import os

router = APIRouter()


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

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
        
         
        # Instantiate LLM
        llm = ChatOpenAI(openai_api_key="sk-proj-kf9yknjQHXCyqASw4jfMT3BlbkFJd3zhvmKFUC414OEaHyZM", model="gpt-3.5-turbo-16k-0613", temperature=0)
        
        # Instantiate SQL agent
        agent_executor = create_sql_agent(llm, toolkit = SQLDatabaseToolkit(db=db, llm=llm), agent_type="openai-tools", verbose=True)
        
        # NL response
        result=agent_executor.invoke({"input":question.text})
        
            
        return {"message":"Success", "data":result}