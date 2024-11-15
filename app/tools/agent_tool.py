# Import abstract syntax grammar
import ast
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from langchain.tools import Tool

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVLY_API_KEY")


### Helper functions ###
# Flattening and filtering truthy values:
def query_as_list(db,query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Generate df from sql query
def generate_dataframe(sql, db):
    df = pd.read_sql(sql, db)
    return df

# interpolate_data
def interpolate_data(df):
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    return df

### Tool ###
# Search tool (built-in support)
tavily_tool = TavilySearchResults(max_results=5)


# API tool (customize tool)
def call_external_api(query=""):
    response = requests.get('https://api.thecatapi.com/v1/images/search', params={'query': query})

    return response.json()

external_api_tool = Tool(
    name="privateAPI",
    func=call_external_api,
    description="Fetches private api"
)


