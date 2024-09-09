from pydantic import BaseModel

class ClassifierResult(BaseModel):
    predicted_class: str