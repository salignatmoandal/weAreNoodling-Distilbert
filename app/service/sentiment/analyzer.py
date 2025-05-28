from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
from app.core.config import settings
from transformers import pipeline
import torch

@dataclass
class SentimentResult:
    label: str
    score: float
    confidence: float
    metadata: Optional[Dict] = None

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
        
    def _load_model(self):
        """Charge le modèle de sentiment analysis."""
        try:
            return pipeline(
                "sentiment-analysis",
                model=settings.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            self.logger.error(f"Erreur de chargement du modèle: {str(e)}")
            raise

    def analyze(self, text: str) -> SentimentResult:
        """Analyse le sentiment d'un texte."""
        try:
            result = self.model(text)[0]
            return SentimentResult(
                label=result["label"].lower(),
                score=result["score"],
                confidence=result["score"],
                metadata={"model": settings.model_name}
            )
        except Exception as e:
            self.logger.error(f"Erreur d'analyse: {str(e)}")
            raise
