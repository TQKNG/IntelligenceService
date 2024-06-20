from fastapi import APIRouter
from app.services.create_agent_service import CreateSqlAgentService, CreateDataAnalysisAgentService
from typing import Dict, Any


# System os and dotenv
import os
from dotenv import load_dotenv
import pandas as pd

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


@router.post("/askdataanalysisagent")
def ask_data_analysis_agent(payload: Dict[Any, Any]):
    question = payload['question']
    if question != "":
        data_analysis_agent = CreateDataAnalysisAgentService()
        data_analysis_agent.create_db_engine(f"mssql+pymssql://{os.getenv("DB_USER")}:{os.getenv("DB_PASS")}@{os.getenv("DB_SERVER")}/{os.getenv("DB_DATABASE")}")

        df = pd.read_sql("Select temperature, enqueuedTime_Stamp from tbl_data WHERE enqueuedTime_Stamp > '2023-06-01' AND enqueuedTime_Stamp <'2023-06-30'", data_analysis_agent.engine)


        data_analysis_agent.config_llm(openai_api_key)
        data_analysis_agent.create_agent(df)

        result = data_analysis_agent.execute(question)

        print("Test result", result)

        return {"message":"Success", "data":result['output']}

