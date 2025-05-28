import re 
import unicodedata
import nltk
import spacy
from nltk.corpus import stopwords
from typing import Optional, List
import logging
from dataclasses import dataclass
from app.core.config import settings

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CleaningConfig:
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = True
    remove_special_chars: bool = True
    to_lowercase: bool = True

class TextCleaner:
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = set(stopwords.words("english"))
            logger.info("TextCleaner initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de TextCleaner: {str(e)}")
            raise

    def _apply_regex_cleaning(self, text: str) -> str:
        """Applique les règles de nettoyage regex."""
        if self.config.remove_urls:
            text = re.sub(r"http\S+|www\S+", "", text)
        if self.config.remove_mentions:
            text = re.sub(r"@\S+", "", text)
        if self.config.remove_hashtags:
            text = re.sub(r"#\S+", "", text)
        if self.config.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

    def clean_text(self, text: str) -> str:
        """Nettoie le texte selon la configuration."""
        if not text:
            return ""
            
        try:
            # Nettoyage de base
            text = self._apply_regex_cleaning(text)
            text = re.sub(r"\s+", " ", text)
            
            # Conversion en minuscules si configuré
            if self.config.to_lowercase:
                text = text.lower()
                
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {str(e)}")
            return text

    def remove_stopwords(self, text: str) -> str:
        """
        Supprime les mots vides du texte.
        
        Args:
            text (str): Le texte à traiter
            
        Returns:
            str: Le texte sans les mots vides
        """
        if not text:
            return ""
            
        try:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            return " ".join(filtered_tokens)
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des mots vides: {str(e)}")
            return text

    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatise le texte.
        
        Args:
            text (str): Le texte à lemmatiser
            
        Returns:
            str: Le texte lemmatisé
        """
        if not text:
            return ""
            
        try:
            doc = self.nlp(text)
            return " ".join([token.lemma_ for token in doc])
        except Exception as e:
            logger.error(f"Erreur lors de la lemmatisation: {str(e)}")
            return text

    def preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte en appliquant toutes les étapes de nettoyage.
        
        Args:
            text (str): Le texte à prétraiter
            
        Returns:
            str: Le texte prétraité
        """
        if not text:
            return ""
            
        try:
            cleaned_text = self.clean_text(text)
            without_stopwords = self.remove_stopwords(cleaned_text)
            lemmatized_text = self.lemmatize_text(without_stopwords)
            return lemmatized_text
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement: {str(e)}")
            return text
