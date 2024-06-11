from langchain_community.utilities import SQLDatabase

class DBService:
    def __init__(self,connection_prop):
       self.db = SQLDatabase.from_uri(f"mssql+pymssql://{connection_prop['database_user']}:{connection_prop['database_password']}@{connection_prop['database_server']}/{connection_prop['database_db']}") 
       
       return self.db