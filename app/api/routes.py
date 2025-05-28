from fastapi import APIRouter, Depends, Header, HTTPException
from app.models.request import SentimentRequest
from app.models.response import GraphSentimentResponse
from app.service.pipeline_sentiment import pipeline_sentiment
from app.core.config import settings

router = APIRouter()

@router.post("/predict", response_model=GraphSentimentResponse)
def predict_sentiment(payload: SentimentRequest, x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    result = pipeline_sentiment(payload.text)
    return result