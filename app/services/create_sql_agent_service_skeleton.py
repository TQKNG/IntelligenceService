import threading
import os
from  app.services.create_agent_service import CreateSqlAgentService
from  app.services.tools.agent_tool import query_as_list
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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
                    CreateSqlAgentServiceSkeleton._instance.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}")
                    CreateSqlAgentServiceSkeleton._instance.config_system_prefix()
                    clients = query_as_list(CreateSqlAgentServiceSkeleton._instance.db,"SELECT DISTINCT client_name, building_name, floor_name, device_name FROM health_data_view")
                    CreateSqlAgentServiceSkeleton._instance.get_client_names(clients)
                    CreateSqlAgentServiceSkeleton._instance.create_custom_retriever_tool(clients)
                    CreateSqlAgentServiceSkeleton._instance.create_example_selector()
                    CreateSqlAgentServiceSkeleton._instance.create_few_shot_prompt()

                  
        return CreateSqlAgentServiceSkeleton._instance