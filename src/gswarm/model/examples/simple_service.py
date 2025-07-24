"""
Simple Example Service Implementation

This example shows a minimal implementation of a custom model service.
"""

import asyncio
from typing import Any, Dict
import numpy as np
from loguru import logger
import sys
import os

# Add parent directory to path to import base_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gswarm.model.base_service import ModelService, ServiceConfig


class SimpleMLService(ModelService):
    """
    A simple machine learning service example.
    
    This service demonstrates the basic pattern for implementing custom services.
    In this example, we simulate a simple linear model.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.model = None
        self.feature_dim = config.extra_args.get("feature_dim", 10)
        
    async def start(self) -> None:
        """Initialize the model"""
        logger.info(f"Initializing simple ML model with {self.feature_dim} features")
        
        # Simulate loading a model (in real case, you'd load from model_path)
        await asyncio.sleep(1)  # Simulate loading time
        
        # Create a simple linear model (random weights for demo)
        self.model = {
            "weights": np.random.randn(self.feature_dim),
            "bias": np.random.randn()
        }
        
        logger.info("Model initialized successfully")
    
    async def stop(self) -> None:
        """Clean up resources"""
        logger.info("Stopping simple ML service")
        self.model = None
    
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        """
        Perform inference on input features.
        
        Args:
            inputs: Feature vector (list or numpy array)
            parameters: Additional parameters (e.g., {"normalize": true})
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Convert inputs to numpy array
        if isinstance(inputs, list):
            features = np.array(inputs)
        elif isinstance(inputs, dict) and "features" in inputs:
            features = np.array(inputs["features"])
        else:
            raise ValueError("Invalid input format. Expected list or dict with 'features' key")
        
        # Validate input shape
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[0]}")
        
        # Normalize if requested
        if parameters.get("normalize", False):
            features = features / (np.linalg.norm(features) + 1e-8)
        
        # Simple linear prediction
        prediction = np.dot(features, self.model["weights"]) + self.model["bias"]
        
        # Apply activation if specified
        activation = parameters.get("activation", None)
        if activation == "sigmoid":
            prediction = 1 / (1 + np.exp(-prediction))
        elif activation == "relu":
            prediction = max(0, prediction)
        
        return {
            "prediction": float(prediction),
            "confidence": float(np.random.rand()),  # Simulated confidence
            "model_version": "1.0.0"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status with model info"""
        status = await super().get_status()
        
        if self.model:
            status.update({
                "model_type": "linear",
                "feature_dimensions": self.feature_dim,
                "model_loaded": True,
            })
        else:
            status.update({
                "model_type": "linear",
                "model_loaded": False,
            })
        
        return status


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ML Model Service")
    parser.add_argument("--name", default="simple-ml", help="Service name")
    parser.add_argument("--port", type=int, default=8080, help="Service port")
    parser.add_argument("--feature-dim", type=int, default=10, help="Feature dimensions")
    parser.add_argument("--model-path", default="dummy.pkl", help="Model path (not used in this example)")
    
    args = parser.parse_args()
    
    config = ServiceConfig(
        name=args.name,
        model_path=args.model_path,
        port=args.port,
        device="cpu",
        extra_args={
            "feature_dim": args.feature_dim
        }
    )
    
    service = SimpleMLService(config)
    service.run()


if __name__ == "__main__":
    main()