from fastapi import FastAPI, Depends
from app.api.routes import router as inference_router
from app.core.config import settings
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis Inference API",
    description="API d'inférence pour l'analyse de sentiment",
    version="1.0.0"
)

# Inclusion des routes d'inférence
app.include_router(
    inference_router,
    prefix="/api/v1/inference",
    tags=["inference"]
)

@app.on_event("startup")
async def startup_event():
    logger.info("Démarrage du service d'inférence...")
    # Initialisation du modèle si nécessaire

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Arrêt du service d'inférence...")
    # Nettoyage des ressources si nécessaire
