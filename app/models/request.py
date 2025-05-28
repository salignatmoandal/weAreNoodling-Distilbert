from pydantic import BaseModel, Field
from typing import Optional

class SentimentRequest(BaseModel):
    text: str = Field(..., example="Create Game – Architecture")
    node_id: Optional[str] = Field(None, example="node_123")
    context: Optional[str] = Field(None, example="This node is part of a creative game design workflow.")

    class Config:
        schema_extra = {
            "example": {
                "text": "Create Game – Architecture",
                "node_id": "node_123",
                "context": "This node is part of a creative game design workflow."
            }
        }