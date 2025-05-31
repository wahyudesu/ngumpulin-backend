"""
Custom exceptions for the application.
This module defines all custom exceptions used throughout the application.
"""

from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class MLPipelineError(HTTPException):
    """Base exception for ML pipeline errors."""
    
    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class ModelNotFoundError(MLPipelineError):
    """Raised when a model cannot be found."""
    
    def __init__(self, model_id: str):
        super().__init__(
            detail=f"Model with ID {model_id} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )

class ModelLoadError(MLPipelineError):
    """Raised when there is an error loading a model."""
    
    def __init__(self, model_path: str):
        super().__init__(
            detail=f"Error loading model from {model_path}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class DataValidationError(MLPipelineError):
    """Raised when data validation fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST
        )

class MLflowError(MLPipelineError):
    """Raised when there is an error with MLflow operations."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) 