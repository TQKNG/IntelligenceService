from fastapi import APIRouter
from app.services.create_agent_service import CreateSqlAgentService, CreateDataAnalysisAgentService
from typing import Dict, Any


# System os and dotenv
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
# SQL Agent Route
@router.post("/asksqlagent")
def ask_sql_agent(payload: Dict[Any,Any]):
    question = payload['question']

    if question != "":
        sql_agent = CreateSqlAgentService()
        sql_agent.config_llm(openai_api_key)
        sql_agent.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}")
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


# Data Analysis Agent Route
@router.post("/askdataanalysisagent")
def ask_data_analysis_agent(payload: Dict[Any, Any]):
    question = payload['question']
    if question != "":
        data_analysis_agent = CreateDataAnalysisAgentService()
        data_analysis_agent.create_db_engine(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}")

        data_analysis_agent.config_system_prefix()
        data_analysis_agent.config_llm(openai_api_key)


        # Test data set: Temperature data in June 2023        
        df = pd.read_sql("SELECT enqueuedTime_Stamp, temperature FROM tbl_data DA INNER JOIN tbl_room R ON DA.roomID = R.id INNER JOIN tbl_floor F ON R.floors_id = F.id WHERE F.id = 1 AND enqueuedTime_Stamp > '2023-06-01' AND enqueuedTime_Stamp <'2023-06-30'ORDER BY enqueuedTime_Stamp ASC", data_analysis_agent.engine)

        # Fill missing values
        # df_filled= df.fillna(method="ffill")
        # print(df_filled.head())

        # Interpolate missing values
        df_interpolated = df.interpolate()
        # print(df_interpolated.head())

        # Plot the data
        df_interpolated['enqueuedTime_Stamp'] = pd.to_datetime(df_interpolated['enqueuedTime_Stamp'])
        
        # df_interpolated.plot(x='enqueuedTime_Stamp', y='temperature', kind='line')
        
        train = df_interpolated[df_interpolated['enqueuedTime_Stamp'] < pd.to_datetime('2023-06-15')]
        test = df_interpolated[df_interpolated['enqueuedTime_Stamp'] >= pd.to_datetime('2023-06-15')]

        # Input
        y = train['temperature']

        # Define ARMA prediction model with SARIMAX class
        ARMAmodel = SARIMAX(y,order=(1,0,1))

        # Fit the model
        ARMAmodel = ARMAmodel.fit()

        # Generate predictions
        forecast_dates = pd.date_range(start=train['enqueuedTime_Stamp'].max(), periods=15, freq='D')
        y_pred = ARMAmodel.get_forecast(steps=15)
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df['Predictions'] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df.index = forecast_dates
        y_pred_out = y_pred_df['Predictions']


        # Plot the data
        # plt.plot(train['enqueuedTime_Stamp'], train['temperature'], label='Train', color='blue')
        # plt.plot(test['enqueuedTime_Stamp'], test['temperature'], label='Test', color='red')
        # plt.plot(test["enqueuedTime_Stamp"],y_pred_out, color='green', label='Predictions')
        # plt.legend()
        # plt.plot(train['enqueuedTime_Stamp'], train['temperature'], label='Train', color='blue')
        # plt.plot(test['enqueuedTime_Stamp'], test['temperature'], label='Test', color='red')
        # plt.show()

        data_analysis_agent.create_agent(y_pred_df)
        result = data_analysis_agent.execute(question)
        return {"message":"Success", "data":result['output']}

        # return{"message":"Success", "data":df.tail(15).to_dict()}


