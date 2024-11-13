import json

my_api_docs ={
    "base_url":"<https://intelligenceservice.azurewebsites.net/",
    "endpoints":{
        "/api/v1/connectagentservice":{
            "method":"GET",
            "description":"Connect to my backend service",
            "parameters":None,
            "response":{
                "description":"A JSON object containing message with status and data with detail message",
                "content_type":"application/json"
            }
        }
    }
}

# Convert the dictionary to a JSON string
my_api_docs = json.dumps(my_api_docs,indent=2)