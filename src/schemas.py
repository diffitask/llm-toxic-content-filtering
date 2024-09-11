from pydantic import BaseModel

class ClassifierInput(BaseModel):
    text: str

class ClassifierOutput(BaseModel):
    """0/1 -- is the prompt/model answer harmful or not"""
    predicted_class: int