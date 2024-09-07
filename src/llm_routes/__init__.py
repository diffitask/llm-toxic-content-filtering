from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_routes.ollama import ollama_route
from llm_routes.zephyr import zephyr_route
from llm_routes.custom_model import custom_model_route

routes = [
    ollama_route,
    # zephyr_route,
    custom_model_route
]

prompt_template = ChatPromptTemplate.from_messages([
    ('user', '{text}')
])

parser = StrOutputParser()

def create_chain(llm):
    chain = prompt_template | llm | parser
    return chain

def configure_models(app):
    for route in routes:
        chain = create_chain(route.llm)
        add_routes(app, chain, path=route.path)