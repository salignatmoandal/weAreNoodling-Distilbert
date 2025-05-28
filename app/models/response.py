# app/models/response.py

from pydantic import BaseModel, Field, constr
from typing import List, Optional, Literal, Dict
from datetime import datetime


class SentimentScore(BaseModel):
    """
    Score de sentiment pour un texte.
    
    Attributes:
        label (Literal): Label du sentiment (positive/negative/neutral)
        score (float): Score de confiance entre 0 et 1
    """
    label: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Label du sentiment"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        example=0.8752,
        description="Score de confiance"
    )


class NodeSentimentResponse(BaseModel):
    """
    Réponse d'analyse de sentiment pour un node.
    
    Attributes:
        node_id (str): Identifiant du node
        sentiment (SentimentScore): Score de sentiment
        metadata (Optional[dict]): Métadonnées supplémentaires
    """
    node_id: str = Field(..., description="Identifiant du node")
    sentiment: SentimentScore
    metadata: Optional[dict] = Field(None, description="Métadonnées supplémentaires")


class EdgeSentimentResponse(BaseModel):
    """
    Réponse d'analyse de sentiment pour un edge.
    
    Attributes:
        edge_id (str): Identifiant de l'edge
        sentiment (SentimentScore): Score de sentiment
        connected_nodes (List[str]): Liste des nodes connectés
        metadata (Optional[dict]): Métadonnées supplémentaires
    """
    edge_id: str = Field(..., description="Identifiant de l'edge")
    sentiment: SentimentScore
    connected_nodes: List[str] = Field(..., description="Liste des nodes connectés")
    metadata: Optional[dict] = Field(None, description="Métadonnées supplémentaires")


class SentimentDistribution(BaseModel):
    """
    Distribution des sentiments dans le graphe.
    
    Attributes:
        positive_nodes (int): Nombre de nodes positifs
        negative_nodes (int): Nombre de nodes négatifs
        positive_edges (int): Nombre d'edges positifs
        negative_edges (int): Nombre d'edges négatifs
    """
    positive_nodes: int = Field(..., ge=0, description="Nombre de nodes positifs")
    negative_nodes: int = Field(..., ge=0, description="Nombre de nodes négatifs")
    positive_edges: int = Field(..., ge=0, description="Nombre d'edges positifs")
    negative_edges: int = Field(..., ge=0, description="Nombre d'edges négatifs")


class GraphSentimentMetrics(BaseModel):
    """
    Métriques globales du graphe.
    
    Attributes:
        average_node_sentiment (float): Score moyen des nodes
        average_edge_sentiment (float): Score moyen des edges
        sentiment_distribution (SentimentDistribution): Distribution des sentiments
    """
    average_node_sentiment: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score moyen des nodes"
    )
    average_edge_sentiment: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score moyen des edges"
    )
    sentiment_distribution: SentimentDistribution


class GraphSentimentResponse(BaseModel):
    """
    Réponse complète d'analyse de sentiment du graphe.
    
    Attributes:
        nodes (List[NodeSentimentResponse]): Analyse des nodes
        edges (List[EdgeSentimentResponse]): Analyse des edges
        metrics (GraphSentimentMetrics): Métriques globales
        metadata (Optional[dict]): Métadonnées supplémentaires
    """
    nodes: List[NodeSentimentResponse] = Field(..., description="Analyse des nodes")
    edges: List[EdgeSentimentResponse] = Field(..., description="Analyse des edges")
    metrics: GraphSentimentMetrics
    metadata: Optional[dict] = Field(
        None,
        description="Métadonnées supplémentaires",
        example={
            "analysis_timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "nodes": [
                    {
                        "node_id": "node1",
                        "sentiment": {
                            "label": "positive",
                            "score": 0.8752
                        },
                        "metadata": {"type": "task"}
                    }
                ],
                "edges": [
                    {
                        "edge_id": "edge1",
                        "sentiment": {
                            "label": "positive",
                            "score": 0.6543
                        },
                        "connected_nodes": ["node1", "node2"],
                        "metadata": {"type": "dependency"}
                    }
                ],
                "metrics": {
                    "average_node_sentiment": 0.8752,
                    "average_edge_sentiment": 0.6543,
                    "sentiment_distribution": {
                        "positive_nodes": 1,
                        "negative_nodes": 0,
                        "positive_edges": 1,
                        "negative_edges": 0
                    }
                },
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_version": "1.0.0"
                }
            }
        }