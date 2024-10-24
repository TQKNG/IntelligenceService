import threading
import os
from  app.services.agent_service import CreateSqlAgentService
from  app.tools.agent_tool import query_as_list
from dotenv import load_dotenv
from app.utils.processing_doc import processing_structured_doc

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlab_api_key = os.getenv("ELEVENLAB_API_KEY")
aai_api_key = os.getenv("AAI_API_KEY")
azure_openai_key = os.getenv("AZURE_OPEN_API_KEY")
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_API_VERSION")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class CreateSqlAgentServiceSkeleton:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if CreateSqlAgentServiceSkeleton._instance is None:
            with CreateSqlAgentServiceSkeleton._lock:
                if CreateSqlAgentServiceSkeleton._instance is None:
                    CreateSqlAgentServiceSkeleton._instance = CreateSqlAgentService()

                    # CreateSqlAgentServiceSkeleton._instance.config_llm(openai_api_key,'gpt-4o-mini')
                    
                    CreateSqlAgentServiceSkeleton._instance.config_llm(azure_openai_key,azure_openai_endpoint, azure_deployment, azure_api_version)
                    print("Connected to OpenAI")
                    

                    CreateSqlAgentServiceSkeleton._instance.config_db(f"mssql+pymssql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_SERVER')}/{os.getenv('DB_DATABASE')}?timeout=3")
                    print("Connected to Database")

                    CreateSqlAgentServiceSkeleton._instance.config_system_prefix()
                    print("Initiate AI instruction")

                    # Retrieve unique client names, building names, floor names and device names
                    clients = query_as_list(CreateSqlAgentServiceSkeleton._instance.db,"SELECT DISTINCT client_name, building_name, floor_name, device_name FROM health_data_view")

                    # Retrieve fields from the table
                    fields = query_as_list(CreateSqlAgentServiceSkeleton._instance.db,"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dbo';")

                    # Set clients 
                    CreateSqlAgentServiceSkeleton._instance.set_client_names(clients)

                    CreateSqlAgentServiceSkeleton._instance.create_custom_retriever_tool(clients + fields)
                    print("Initiate Retriever Tool")
                    
                    docs_to_object = processing_structured_doc()
                    

                    CreateSqlAgentServiceSkeleton._instance.create_document_retriever_tool(docs_to_object)

                    CreateSqlAgentServiceSkeleton._instance.create_example_selector()
                    print("Initiate Example Selector Tool")
                    
                    CreateSqlAgentServiceSkeleton._instance.create_few_shot_prompt()
                    print("Initiate Few Shots Prompt")

                  
        return CreateSqlAgentServiceSkeleton._instance