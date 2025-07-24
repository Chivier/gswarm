"""
Example model service implementations
"""

from .simple_service import SimpleMLService
from .vllm_service import VLLMService

__all__ = ["SimpleMLService", "VLLMService"]