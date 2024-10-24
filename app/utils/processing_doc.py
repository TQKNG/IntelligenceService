import os
import pandas as pd

# Import document loader
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

# Text splitter
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain_community.document_loaders.csv_loader import CSVLoader


# Step 1: Pre-process excel data
def excel_to_csv(file_path=None):
    # Path to file
    directory = './uploads'
    excel_file_path = os.path.join(directory, "WELL-Standard-Summary.xlsx")
    csv_file_path = os.path.join(directory,"WELL-Standard-Summary.csv")
    df = pd.read_excel(excel_file_path)
    df.to_csv(csv_file_path, index = False)

    


# Step 2: Processing structure docs eg: excel
def processing_structured_doc( filepath=None, endpoint=None, api_key=None):
    # Path to file
    directory = './uploads'
    csv_file_path = os.path.join(directory,"WELL-Standard-Summary.csv")

    # Step 1
    excel_to_csv()

    loader = CSVLoader(file_path=csv_file_path, source_column="Source")
    data = loader.load()

    return data

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

processing_structured_doc()
