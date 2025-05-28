# app/models/response.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SentimentScore(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(..., ge=0.0, le=1.0, example=0.8752)


class NodeSentimentResponse(BaseModel):
    node_id: str
    sentiment: SentimentScore
    metadata: Optional[dict] = None


class EdgeSentimentResponse(BaseModel):
    edge_id: str
    sentiment: SentimentScore
    connected_nodes: List[str]
    metadata: Optional[dict] = None


class GraphSentimentMetrics(BaseModel):
    average_node_sentiment: float
    average_edge_sentiment: float
    sentiment_distribution: dict


class GraphSentimentResponse(BaseModel):
    nodes: List[NodeSentimentResponse]
    edges: List[EdgeSentimentResponse]
    metrics: GraphSentimentMetrics