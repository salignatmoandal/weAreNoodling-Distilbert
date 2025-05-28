from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Vérifie la clé API."""
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Clé API invalide"
        )
    return api_key
