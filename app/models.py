from  pydantic import BaseModel

# Question Model
class Question(BaseModel):
    text:str

# Answer Model
class Answer(BaseModel):
    text:str