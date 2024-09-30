from pydantic import BaseModel

# class ClassifierInput(BaseModel):
#     text: str

class ClassifierOutput(BaseModel):
    "User prompt or LLM text answer to the user request"
    input: str

    """0/1 -- is the prompt/model answer harmful or not"""
    # has_jailbreak: str
    predicted_class: str