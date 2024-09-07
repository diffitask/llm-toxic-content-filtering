from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

class BaseLLMRoute(BaseModel):
    llm: BaseChatModel
    path: str