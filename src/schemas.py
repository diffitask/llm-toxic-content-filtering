from pydantic import BaseModel

class ClassifierInput(BaseModel):
    text: str

class ClassifierOutput(BaseModel):
    predicted_class: str