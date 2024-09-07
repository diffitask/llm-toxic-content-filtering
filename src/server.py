#!/usr/bin/env python
from fastapi import FastAPI, Request
from llm_routes import configure_models
from middlewares.toxic_filtering import ToxicFiltering
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(
    title="LLM Toxic Content Filtering",
    version=1.0
)

configure_models(app)

# @app.middleware("toxic_filtering")
# async def add_process_time_header(request: Request, call_next):
#     print(await request.body())
#     response = await call_next(request)
#     print(response)
#     return response

app.add_middleware(ToxicFiltering)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)