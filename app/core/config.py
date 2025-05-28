from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Configuration du modèle
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    api_key: str = "API_KEY"
    
    # Configuration du prétraitement
    max_text_length: int = 512
    language: str = "english"
    
    # Configuration de l'API
    api_version: str = "v1"
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
