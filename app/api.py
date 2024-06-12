from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers import router as controllers

app = FastAPI()

origins =[
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to Intelligence Service API"}

# Other routes api/v1
app.include_router(controllers, prefix="/api/v1", tags=["controllers"])