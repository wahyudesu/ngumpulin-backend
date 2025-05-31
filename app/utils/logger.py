"""
Logging configuration for the application.
This module sets up structured logging using structlog.
"""

import structlog
import logging
import sys
from typing import Any, Dict

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: The logging level to use (default: INFO)
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A configured logger instance
    """
    return structlog.get_logger(name)

# Initialize default logger
logger = get_logger(__name__) 