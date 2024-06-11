from fastapi import APIRouter
from app.models import Question, Answer

router = APIRouter()

@router.post("/askAgent")
def ask_agent(question:Question):
    if question.text != "":
        return Answer(text= "I am an AI assistant and I am here to help you")