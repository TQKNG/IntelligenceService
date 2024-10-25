import os
import pandas as pd

# Import document loader
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

# Text splitter
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain_community.document_loaders.csv_loader import CSVLoader



async def save_file(file, file_path):
    with open(file_path, 'wb') as f:
        f.write(await file.read())
    
async def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


# Pre-process excel data
async def excel_to_csv(file_path):
    # Path to file
    df = pd.read_excel(file_path)
    df.to_csv(file_path, index = False)

    

# Processing structure docs eg: excel
async def processing_structured_doc(file_path):

    # Step 1
    await excel_to_csv(file_path)

    loader = CSVLoader(file_path=file_path, source_column="Source")
    data = loader.load()

    return data

def processing_unstructured_doc(file_path):
    pass

    # # Load Data
    # loader = AzureAIDocumentIntelligenceLoader(
    # api_endpoint="https://intelligence-document.cognitiveservices.azure.com/",
    # api_key="37f5c66594fb420b95d73ee3543b2b24",
    # file_path= csv_file_path,
    # api_model="prebuilt-layout")

    # # Text splitter and chunk
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100,  separators=[",", "\n", " ", "\t", ""],length_function=len,is_separator_regex=False)

    # # Split data
    # document = loader.load()
    # texts = text_splitter.split_documents(documents=document)

    # print("test my text", texts)

