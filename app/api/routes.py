from fastapi import APIRouter, Depends, HTTPException
from app.models.request import SentimentRequest
from app.models.response import GraphSentimentResponse
from app.service.pipeline_sentiment import SentimentAnalyzer
from app.core.security import verify_api_key
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Dépendance pour l'analyseur de sentiment
def get_sentiment_analyzer():
    return SentimentAnalyzer()

@router.post("/predict", response_model=GraphSentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint d'inférence pour l'analyse de sentiment.
    
    Args:
        request (SentimentRequest): Requête contenant le texte à analyser
        analyzer (SentimentAnalyzer): Instance de l'analyseur de sentiment
        api_key (str): Clé API pour l'authentification
        
    Returns:
        GraphSentimentResponse: Résultat de l'inférence
    """
    try:
        logger.info(f"Inférence demandée pour le texte: {request.text[:50]}...")
        
        # Inférence pour un node spécifique
        if request.node_id:
            result = analyzer.analyze_node({
                "id": request.node_id,
                "text": request.text,
                "metadata": {
                    "context": request.context,
                    "inference_type": "node_sentiment"
                }
            })
            return GraphSentimentResponse(
                nodes=[result],
                edges=[],
                metrics={
                    "average_node_sentiment": result["sentiment"]["score"],
                    "average_edge_sentiment": 0.0,
                    "sentiment_distribution": {
                        "positive_nodes": 1 if result["sentiment"]["label"] == "positive" else 0,
                        "negative_nodes": 1 if result["sentiment"]["label"] == "negative" else 0,
                        "positive_edges": 0,
                        "negative_edges": 0
                    }
                }
            )
        
        # Inférence simple du texte
        result = analyzer.analyze_text(request.text)
        return GraphSentimentResponse(
            nodes=[{
                "node_id": "temp_node",
                "sentiment": result,
                "metadata": {
                    "context": request.context,
                    "inference_type": "text_sentiment"
                }
            }],
            edges=[],
            metrics={
                "average_node_sentiment": result["score"],
                "average_edge_sentiment": 0.0,
                "sentiment_distribution": {
                    "positive_nodes": 1 if result["label"] == "positive" else 0,
                    "negative_nodes": 1 if result["label"] == "negative" else 0,
                    "positive_edges": 0,
                    "negative_edges": 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour la santé du modèle
@router.get("/health")
async def model_health(analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)):
    """
    Vérifie la santé du modèle d'inférence.
    """
    try:
        # Test simple pour vérifier que le modèle fonctionne
        test_result = analyzer.analyze_text("Test health check")
        return {
            "status": "healthy",
            "model": settings.model_name,
            "test_inference": test_result
        }
    except Exception as e:
        logger.error(f"Erreur de santé du modèle: {str(e)}")
        raise HTTPException(status_code=503, detail="Model not healthy")