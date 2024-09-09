from pydantic import BaseModel

class UserPrompt(BaseModel):
    id: int
    prompt: str