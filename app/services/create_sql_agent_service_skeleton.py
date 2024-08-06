import threading
import os
from  app.services.create_agent_service import CreateSqlAgentService
from app.services.real_time_voice_service import AI_Assistant
from  app.services.tools.agent_tool import query_as_list
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlab_api_key = os.getenv("ELEVENLAB_API_KEY")
aai_api_key = os.getenv("AAI_API_KEY")

class CreateSqlAgentServiceSkeleton:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if CreateSqlAgentServiceSkeleton._instance is None:
            with CreateSqlAgentServiceSkeleton._lock:
                if CreateSqlAgentServiceSkeleton._instance is None:
                    CreateSqlAgentServiceSkeleton._instance = CreateSqlAgentService()

                    CreateSqlAgentServiceSkeleton._instance.config_llm(openai_api_key)
                    print("Connected to OpenAI")

                    CreateSqlAgentServiceSkeleton._instance.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}?timeout=3")
                    print("Connected to Database")

                    CreateSqlAgentServiceSkeleton._instance.config_system_prefix()
                    clients = query_as_list(CreateSqlAgentServiceSkeleton._instance.db,"SELECT DISTINCT client_name, building_name, floor_name, device_name FROM health_data_view")
                    fields = query_as_list(CreateSqlAgentServiceSkeleton._instance.db,"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dbo';")

                    CreateSqlAgentServiceSkeleton._instance.get_client_names(clients)
                    CreateSqlAgentServiceSkeleton._instance.create_custom_retriever_tool(clients + fields)
                    CreateSqlAgentServiceSkeleton._instance.create_example_selector()
                    CreateSqlAgentServiceSkeleton._instance.create_few_shot_prompt()
                    # CreateSqlAgentServiceSkeleton._instance = AI_Assistant()

                  
        return CreateSqlAgentServiceSkeleton._instance