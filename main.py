from fastapi import FastAPI
import uvicorn
import os
os.environ["LANGCHAIN_TRACING"] = "true"


# Instantatiate FastAPI instance
app = FastAPI()

# Main
if __name__ == "__main__":
    uvicorn.run("app.routes.routes:app", host="0.0.0.0", port=8000, reload=True)

