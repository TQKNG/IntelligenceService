# Import abstract syntax grammar
import ast
import re
import pandas as pd
import matplotlib.pyplot as plt

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


# Search tool


