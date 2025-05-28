from pydantic import BaseModel, Field, constr
from typing import Optional
from datetime import datetime

class SentimentRequest(BaseModel):
    """
    Modèle de requête pour l'analyse de sentiment.
    
    Attributes:
        text (str): Le texte à analyser
        node_id (Optional[str]): Identifiant du node (optionnel)
        context (Optional[str]): Contexte supplémentaire (optionnel)
        metadata (Optional[dict]): Métadonnées supplémentaires (optionnel)
    """
    text: constr(min_length=1, max_length=1000) = Field(
        ...,
        example="Create Game – Architecture",
        description="Texte à analyser"
    )
    node_id: Optional[str] = Field(
        None,
        example="node_123",
        description="Identifiant du node"
    )
    context: Optional[str] = Field(
        None,
        example="This node is part of a creative game design workflow.",
        description="Contexte supplémentaire"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Métadonnées supplémentaires"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "Create Game – Architecture",
                "node_id": "node_123",
                "context": "This node is part of a creative game design workflow.",
                "metadata": {
                    "source": "api",
                    "timestamp": datetime.now().isoformat()
                }
            }
        }