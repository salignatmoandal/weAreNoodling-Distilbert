from pydantic import BaseModel, Field, constr
from typing import Optional
from datetime import datetime

class SentimentRequest(BaseModel):
    """
    Request model for sentiment analysis.

    Attributes:
        text (str): The text to analyze.
        node_id (Optional[str]): Node identifier (optional).
        context (Optional[str]): Additional context for analysis (optional).
        metadata (Optional[dict]): Extra metadata (optional).
    """
    text: constr(min_length=1, max_length=1000) = Field(
        ...,
        example="Create Game – Architecture",
        description="The text to be analyzed"
    )
    node_id: Optional[str] = Field(
        None,
        example="node_123",
        description="Node identifier"
    )
    context: Optional[str] = Field(
        None,
        example="This node is part of a creative game design workflow.",
        description="Additional context for analysis"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Extra metadata (e.g., source, timestamp, tags)"
    )

    class Config:
        # Example schema for OpenAPI docs and validation previews
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
