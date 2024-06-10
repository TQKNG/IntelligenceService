from fastapi import FastAPI

# Instantatiate FastAPI instance
app = FastAPI()


# Connect to database on startup and disconnect on shutdown

# @app.on_event("startup")
# async def startup():
#     await database.connect()

# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()

app.include_router(aiAgent, prefix='/api/v1/aiAgent', tags=['aiAgent'])