from fastapi import FastAPI
import uvicorn
from app.controllers import router as controllers

# Instantatiate FastAPI instance
app = FastAPI()

# Main
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

