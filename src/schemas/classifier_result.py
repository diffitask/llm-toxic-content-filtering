from pydantic import BaseModel

class ClassifierResult(BaseModel):
    toxic: bool