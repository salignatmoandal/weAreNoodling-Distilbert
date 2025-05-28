from pydantic import BaseSettings

class Settings(BaseSettings):
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    api_key: str = "API_KEY"

    class Config:
        env_file = ".env"

settings = Settings()
