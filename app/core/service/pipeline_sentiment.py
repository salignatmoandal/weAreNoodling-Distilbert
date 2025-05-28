from transformers import pipeline
from functools import lru_cache
from app.core.config import settings

@lru_cache()
def get_model():
    return pipeline("sentiment-analysis", model=settings.model_name, token=settings.api_key)

def analyze_sentiment(text: str):
    model = get_model()
    result = model(text)[0]
    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 4),
    }