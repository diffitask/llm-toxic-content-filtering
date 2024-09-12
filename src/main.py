from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from src.classifiers import configure_classifiers
from src.alerts import configure_alerts

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

classifier_router = APIRouter(prefix="/classifiers")
alert_router = APIRouter(prefix="/alerts")

configure_classifiers(classifier_router)
configure_alerts(alert_router)

app.include_router(classifier_router)
app.include_router(alert_router)