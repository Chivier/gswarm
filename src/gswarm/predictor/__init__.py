"""
GSwarm Predictor Module
"""

from gswarm.predictor.predicting import (
    PredictorOutput,
    predict_inference,
    update_model_predictor,
)
from gswarm.predictor.simple_predictor import (
    Predictor,
    predictor,
)

__all__ = [
    # Original API
    "PredictorOutput",
    "predict_inference",
    "update_model_predictor",
    # Simple API
    "Predictor",
    "predictor",
]