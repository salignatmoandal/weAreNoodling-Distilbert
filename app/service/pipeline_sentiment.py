from typing import Dict, List, Optional, Union
from transformers import pipeline
from functools import lru_cache
import numpy as np
import logging
from app.core.config import settings
from app.utils.text_cleaner import preprocess_text

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyseur de sentiment pour les textes, nodes et edges d'un graphe."""

    def __init__(self):
        """Initialise l'analyseur de sentiment avec le modèle configuré."""
        self.model = self._load_model()
        logger.info("SentimentAnalyzer initialisé avec succès")

    @lru_cache()
    def _load_model(self):
        """Charge le modèle de sentiment analysis avec mise en cache."""
        try:
            return pipeline(
                "sentiment-analysis", 
                model=settings.model_name, 
                token=settings.api_key
            )
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyse le sentiment d'un texte donné.
        
        Args:
            text (str): Le texte à analyser
            
        Returns:
            Dict[str, Union[str, float]]: Résultat de l'analyse avec label et score
        """
        try:
            cleaned_text = preprocess_text(text)
            result = self.model(cleaned_text)[0]
            return {
                "label": result["label"].lower(),
                "score": round(result["score"], 4)
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du texte: {str(e)}")
            raise

    def analyze_node(self, node_data: Dict) -> Dict:
        """
        Analyse le sentiment d'un node.
        
        Args:
            node_data (Dict): Données du node contenant au minimum 'id' et 'text'
            
        Returns:
            Dict: Résultat de l'analyse avec les métadonnées du node
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
            logger.error(f"Erreur lors de l'analyse du node: {str(e)}")
            raise

    def analyze_edge(self, edge_data: Dict, connected_nodes: List[Dict]) -> Dict:
        """
        Analyse le sentiment d'un edge basé sur les nodes connectés.
        
        Args:
            edge_data (Dict): Données de l'edge
            connected_nodes (List[Dict]): Liste des nodes connectés
            
        Returns:
            Dict: Résultat de l'analyse avec les métadonnées de l'edge
        """
        try:
            # Calculer le sentiment agrégé des nodes connectés
            node_sentiments = [
                self.analyze_node(node)["sentiment"]["score"] 
                for node in connected_nodes
            ]
            
            # Calculer la moyenne des scores de sentiment
            avg_sentiment = np.mean(node_sentiments)
            
            # Déterminer le label basé sur le score moyen
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
            logger.error(f"Erreur lors de l'analyse de l'edge: {str(e)}")
            raise

    def analyze_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """
        Analyse le sentiment de l'ensemble du graphe.
        
        Args:
            nodes (List[Dict]): Liste des nodes du graphe
            edges (List[Dict]): Liste des edges du graphe
            
        Returns:
            Dict: Résultat complet de l'analyse avec métriques
        """
        try:
            # Analyser tous les nodes
            node_analyses = [self.analyze_node(node) for node in nodes]
            
            # Créer un dictionnaire des nodes pour accès rapide
            nodes_dict = {node["id"]: node for node in nodes}
            
            # Analyser tous les edges
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
            
            # Calculer les métriques globales
            node_sentiments = [analysis["sentiment"]["score"] for analysis in node_analyses]
            edge_sentiments = [analysis["sentiment"]["score"] for analysis in edge_analyses]
            
            return {
                "nodes": node_analyses,
                "edges": edge_analyses,
                "metrics": {
                    "average_node_sentiment": round(np.mean(node_sentiments), 4),
                    "average_edge_sentiment": round(np.mean(edge_sentiments), 4),
                    "sentiment_distribution": {
                        "positive_nodes": sum(1 for n in node_analyses if n["sentiment"]["label"] == "positive"),
                        "negative_nodes": sum(1 for n in node_analyses if n["sentiment"]["label"] == "negative"),
                        "positive_edges": sum(1 for e in edge_analyses if e["sentiment"]["label"] == "positive"),
                        "negative_edges": sum(1 for e in edge_analyses if e["sentiment"]["label"] == "negative")
                    }
                }
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du graphe: {str(e)}")
            raise