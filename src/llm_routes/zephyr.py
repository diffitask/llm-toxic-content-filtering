from llm_routes.base import BaseLLMRoute
from langchain_community.chat_models import ChatOllama

zephyr_route = BaseLLMRoute(
    llm=ChatOllama(model="zephyr"),
    path="/zephyr"
)