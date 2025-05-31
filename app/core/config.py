"""
Configuration settings for the FastAPI application.
This module handles all environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Ngumpulin Backend"
    
    # Database Settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # ML Settings
    PINECONE_API_KEY: str
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MODEL_PATH: str = "models/best_model.pkl"
    
    # Security Settings
    SECRET_KEY: str = os.urandom(32).hex()
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 