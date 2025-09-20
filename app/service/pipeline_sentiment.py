from typing import Dict, List, Union
from transformers import pipeline
from functools import lru_cache
import numpy as np
import logging
from app.core.config import settings
from app.utils.text_cleaner import preprocess_text

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Perform sentiment analysis on text, nodes, edges, and entire graphs."""

    def __init__(self):
        """Initialize the sentiment analyzer with the configured model."""
        self.model = self._load_model()
        logger.info("SentimentAnalyzer successfully initialized")

    @lru_cache()
    def _load_model(self):
        """
        Load and cache the sentiment analysis model.

        Returns:
            HuggingFace pipeline object for sentiment analysis.
        """
        try:
            return pipeline(
                "sentiment-analysis", 
                model=settings.model_name, 
                token=settings.api_key  # ⚠️ Not a standard HuggingFace param, check config
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text string.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, Union[str, float]]: Sentiment result with label and score.
        """
        try:
            cleaned_text = preprocess_text(text)  # Clean input before inference
            result = self.model(cleaned_text)[0]  # HuggingFace returns a list, take first item
            return {
                "label": result["label"].lower(),   # Normalize label to lowercase
                "score": round(result["score"], 4)  # Round score to 4 decimals
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise

    def analyze_node(self, node_data: Dict) -> Dict:
        """
        Analyze sentiment of a graph node.

        Args:
            node_data (Dict): Node data containing at least 'id' and 'text'.

        Returns:
            Dict: Sentiment result including node metadata.
        """
        try:
            text = node_data.get("text", "")
            sentiment = self.analyze_text(text)
            return {
                "node_id": node_data.get("id"),
                "sentiment": sentiment,
                "metadata": node_data.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error analyzing node: {str(e)}")
            raise

    def analyze_edge(self, edge_data: Dict, connected_nodes: List[Dict]) -> Dict:
        """
        Analyze sentiment of an edge based on its connected nodes.

        Args:
            edge_data (Dict): Edge information (id, metadata, etc.)
            connected_nodes (List[Dict]): List of connected nodes.

        Returns:
            Dict: Sentiment result for the edge with aggregated scores.
        """
        try:
            # Collect sentiment scores from connected nodes
            node_sentiments = [
                self.analyze_node(node)["sentiment"]["score"] 
                for node in connected_nodes
            ]
            
            # Compute average sentiment across connected nodes
            avg_sentiment = np.mean(node_sentiments)
            
            # Assign label based on threshold
            label = "positive" if avg_sentiment > 0.5 else "negative"
            
            return {
                "edge_id": edge_data.get("id"),
                "sentiment": {
                    "label": label,
                    "score": round(avg_sentiment, 4)
                },
                "connected_nodes": [node.get("id") for node in connected_nodes],
                "metadata": edge_data.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error analyzing edge: {str(e)}")
            raise

    def analyze_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """
        Perform sentiment analysis on an entire graph.

        Args:
            nodes (List[Dict]): List of graph nodes.
            edges (List[Dict]): List of graph edges.

        Returns:
            Dict: Complete analysis including node/edge results and metrics.
        """
        try:
            # Analyze all nodes individually
            node_analyses = [self.analyze_node(node) for node in nodes]
            
            # Build dictionary for fast node lookup by id
            nodes_dict = {node["id"]: node for node in nodes}
            
            # Analyze all edges
            edge_analyses = []
            for edge in edges:
                source_node = nodes_dict.get(edge.get("source"))
                target_node = nodes_dict.get(edge.get("target"))
                if source_node and target_node:
                    edge_analysis = self.analyze_edge(
                        edge, 
                        [source_node, target_node]
                    )
                    edge_analyses.append(edge_analysis)
            
            # Compute global metrics
            node_sentiments = [analysis["sentiment"]["score"] for analysis in node_analyses]
            edge_sentiments = [analysis["sentiment"]["score"] for analysis in edge_analyses]
            
            return {
                "nodes": node_analyses,
                "edges": edge_analyses,
                "metrics": {
                    "average_node_sentiment": round(np.mean(node_sentiments), 4),
                    "average_edge_sentiment": round(np.mean(edg
