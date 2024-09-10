from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from src.classifiers import configure_classifiers
import uvicorn

load_dotenv()

app = FastAPI(
    title="LLM Toxic Content Filtering",
    version=1.0
)

@app.get('/')
async def root():
    return RedirectResponse('/docs')

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

configure_classifiers(app)