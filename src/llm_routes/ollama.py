from llm_routes.base import BaseLLMRoute
from langchain_community.chat_models import ChatOllama

ollama_route = BaseLLMRoute(
    llm=ChatOllama(model="llama3"),
    path="/llama3"
)