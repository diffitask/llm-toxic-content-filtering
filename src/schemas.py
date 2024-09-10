from pydantic import BaseModel
import pydantic

print(pydantic.__version__)

class ClassifierInput(BaseModel):
    text: str

class ClassifierOutput(BaseModel):
    predicted_class: str