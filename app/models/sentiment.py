from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

from app.core.config import Settings

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=Settings.max_text_length)
    metadata: Optional[Dict] = Field(default=None)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text for sentiment analysis",
                "metadata": {"source": "api", "timestamp": datetime.now().isoformat()}
            }
        }

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., regex="^(positive|negative|neutral)$")
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict] = None
