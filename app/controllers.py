from fastapi import APIRouter, HTTPException, requests
from fastapi.responses import StreamingResponse,FileResponse
from app.services.create_agent_service import CreateSqlAgentService, CreateDataAnalysisAgentService
from app.services.real_time_voice_service import AI_Assistant
from typing import Dict, Any
import requests
import base64

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
elevenlab_api_key = os.getenv("ELEVENLAB_API_KEY")
aai_api_key = os.getenv("AAI_API_KEY")

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
async def ask_sql_agent(payload: Dict[Any,Any]):
    question = payload['question']

    if question != "":
        sql_agent = CreateSqlAgentService()
        sql_agent.config_llm(openai_api_key)
        sql_agent.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}")
        sql_agent.config_system_prefix()

        ### Turn data from each table to list of keywords ###
        clients = query_as_list(sql_agent.db,"SELECT client_name FROM health_data_view")
        columns = query_as_list(sql_agent.db,"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'health_data_view';")

        ### Create custom retriever tool ###
        sql_agent.create_custom_retriever_tool(clients)

        sql_agent.create_example_selector()
        sql_agent.create_few_shot_prompt()
        sql_agent.create_full_prompt(question)
        sql_agent.create_agent()

        ## Execute the agent and return the response
        # result = sql_agent.execute(question)

        ## Stream the response in server
        # async for event in sql_agent.agent.astream_events(question, version="v1"):
        #     if event['event'] == 'on_chat_model_start':
        #         print("Chat model started", flush=True)
        #     elif event['event'] == 'on_chat_model_stream':
        #         print(event['data']['chunk'].content, end="", flush=True)

        ## Stream the response through API
        async def generate_chat_response(message):
            async for chunk in sql_agent.agent.astream(question):
                content = chunk
                
                # Separate the steps, actions and final output
                for msg_type in content:
                    if msg_type == "output":
                        yield f"{chunk['output']}\n\n"
                    elif msg_type == "actions":
                        yield f"{chunk['actions'][0]}\n\n"
                    else:
                        yield f"Further processing\n\n"

        return StreamingResponse(generate_chat_response(message=question), media_type="text/event-stream")
        # return {"message":"Success", "data":"done"}


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
        plt.figure(figsize=(10, 6))
        # plt.plot(train['enqueuedTime_Stamp'], train['temperature'], label='Train', color='blue')
        plt.plot(test['enqueuedTime_Stamp'], test['temperature'], label='Test', color='red')
        # plt.plot(test["enqueuedTime_Stamp"],y_pred_out, color='green', label='Predictions')
        plt.legend()
        output_dir = 'C:/temp/data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'temperature_plot.png')
        plt.savefig(output_path)
        plt.close()
        # plt.plot(train['enqueuedTime_Stamp'], train['temperature'], label='Train', color='blue')
        # plt.plot(test['enqueuedTime_Stamp'], test['temperature'], label='Test', color='red')
        # plt.show()

        data_analysis_agent.create_agent(y_pred_df)
        result = data_analysis_agent.execute(question)

        return {"message":"Success", "data":result['output'], "plot_url":'http://127.0.0.1:8000/api/v1/plot'}

        # return{"message":"Success", "data":df.tail(15).to_dict()}

@router.get("/plot")
def serve_plot():
    return FileResponse('C:/temp/data/temperature_plot.png', media_type='image/png')

@router.get("/test-voice")
def text_to_speech():
    assistance_agent = AI_Assistant()
    text = "Hello, How can I help you?"
    audio_stream = assistance_agent.text_to_speech(text)
    return audio_stream
     
@router.post("/test-voice")
def speech_to_text(payload: Dict[Any, Any]):
    assistance_agent = AI_Assistant()

    audio_base64 = payload['audio']
    audio_bytes = base64.b64decode(audio_base64)

    file_path = 'uploads/audio.mp3'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_bytes)

    text = assistance_agent.speech_to_text(file_path)

    # if not text:
    #     raise ValueError("Text extraction failed or returned empty. Ensure the audio file is valid and contains recognizable speech.")
    
    # sql_agent = CreateSqlAgentService()
    # sql_agent.config_llm(openai_api_key)
    # sql_agent.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}")
    # sql_agent.config_system_prefix()

    # ### Turn data from each table to list of keywords ###
    # clients = query_as_list(sql_agent.db,"SELECT client_name FROM health_data_view")
    # columns = query_as_list(sql_agent.db,"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'health_data_view';")

    # ### Create custom retriever tool ###
    # sql_agent.create_custom_retriever_tool(clients)

    # sql_agent.create_example_selector()
    # sql_agent.create_few_shot_prompt()
    # sql_agent.create_full_prompt(text)
    # sql_agent.create_agent()

    # result = sql_agent.execute(text)

    # print("Test my result", result)

    ai_response = assistance_agent.generate_openai_response(text)

    audio_stream = assistance_agent.text_to_speech(ai_response)
    
    return audio_stream