import time
import pytest
from app.utils.text_cleaner import TextCleaner

class PerformanceMetrics:
    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Mesure le temps d'exécution d'une fonction."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

def test_processing_speed(text_cleaner):
    """Test la vitesse de traitement d'un texte long."""
    # Création d'un texte long pour le test
    long_text = "This is a test " * 1000
    
    # Mesure du temps de traitement
    _, processing_time = PerformanceMetrics.measure_time(
        text_cleaner.preprocess_text, 
        long_text
    )
    
    # Vérification que le traitement est rapide
    assert processing_time < 5.0, f"Le traitement a pris {processing_time:.2f} secondes, ce qui est trop long"

def test_batch_processing(text_cleaner):
    """Test le traitement par lots de plusieurs textes."""
    # Création d'un ensemble de textes de test
    test_texts = [
        "Hello! This is a test #hashtag @mention http://example.com",
        "RT @user: The quick brown fox #jumps over the lazy dog!",
        "This is a test with émojis 😊 and special chars @#$%",
        "the quick brown fox jumps over the lazy dog",
        "running jumping swimming"
    ] * 20  # Répéter 20 fois pour un test significatif
    
    # Mesure du temps de traitement par lots
    start_time = time.time()
    results = []
    for text in test_texts:
        result = text_cleaner.preprocess_text(text)
        results.append(result)
    total_time = time.time() - start_time
    
    # Vérifications
    assert len(results) == len(test_texts), "Tous les textes n'ont pas été traités"
    assert total_time < 10.0, f"Le traitement par lots a pris {total_time:.2f} secondes, ce qui est trop long"
    
    # Vérification de la qualité du traitement
    for result in results:
        assert isinstance(result, str), "Le résultat n'est pas une chaîne de caractères"
        assert result.islower(), "Le texte n'est pas en minuscules"
        assert "http" not in result, "Les URLs n'ont pas été supprimées"
        assert "@" not in result, "Les mentions n'ont pas été supprimées"
        assert "#" not in result, "Les hashtags n'ont pas été supprimés"

def test_memory_efficiency(text_cleaner):
    """Test l'efficacité mémoire en traitant un grand nombre de textes."""
    # Création d'un grand nombre de textes
    test_texts = ["Sample text for memory test"] * 1000
    
    # Traitement des textes
    results = []
    for text in test_texts:
        result = text_cleaner.preprocess_text(text)
        results.append(result)
    
    # Vérification que tous les textes ont été traités correctement
    assert len(results) == len(test_texts), "Tous les textes n'ont pas été traités"
    assert all(isinstance(r, str) for r in results), "Certains résultats ne sont pas des chaînes de caractères"
    assert all(r == "sample text for memory test" for r in results), "Les textes n'ont pas été traités correctement"

def test_error_handling(text_cleaner):
    """Test la gestion des erreurs avec des entrées invalides."""
    # Test avec None
    result = text_cleaner.preprocess_text(None)
    assert result == "", "Le texte None n'a pas été géré correctement"
    
    # Test avec un nombre
    result = text_cleaner.preprocess_text(123)
    assert result == "", "Le nombre n'a pas été géré correctement"
    
    # Test avec une liste
    result = text_cleaner.preprocess_text(["test"])
    assert result == "", "La liste n'a pas été gérée correctement"
