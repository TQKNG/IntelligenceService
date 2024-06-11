from fastapi import FastAPI
from app.controllers import router as controllers

# Instantatiate FastAPI instance
app = FastAPI()




# Connect to database on startup and disconnect on shutdown

# @app.on_event("startup")
# async def startup():
#     await database.connect()

# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()

# Define a route
app.include_router(controllers, prefix="/api/v1", tags=["controllers"])