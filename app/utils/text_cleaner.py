import re 
import unicodedata
import nltk
import spacy
from nltk.corpus import stopwords
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self, language: str = "english"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = set(stopwords.words(language))
            logger.info(f"TextCleaner initialisé avec succès pour la langue {language}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de TextCleaner: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte en supprimant les caractères spéciaux et les URLs.
        
        Args:
            text (str): Le texte à nettoyer
            
        Returns:
            str: Le texte nettoyé
        """
        if not text:
            return ""
            
        try:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"RT", "", text)
    text = re.sub(r"via", "", text)
    text = re.sub(r"amp", "", text)
    text = re.sub(r"&amp;", "", text)
    return text.strip().lower()
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du texte: {str(e)}")
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
