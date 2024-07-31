from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.controllers import router as controllers
from app.services.real_time_voice_service import AI_Assistant

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
    expose_headers=["Access-Control-Allow-Origin"]
)

# Root route
@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to Intelligence Service API"}

# @app.websocket("/ws/voicestream")
# async def voice_stream(websocket: WebSocket):
#     voice_agent = AI_Assistant()
#     greeting = "Thank you for using Virbrix Analytic assistant. My name is Virbrix. How can I help you today?"
#     await websocket.accept()

#     try:
#         while True:
#             data = await websocket.receive_text()
#             print(f"Received message: {data}")

#             # Generate audio stream 
#             audio_stream =  voice_agent.generate_audio_stream(
#                 text=greeting
#             )

#             # Send audio stream
#             async for chunk in audio_stream:
#                 await websocket.send_bytes(chunk)
                
#     except WebSocketDisconnect:
#         print("Client disconnected")

# Other routes api/v1
app.include_router(controllers, prefix="/api/v1", tags=["controllers"])