"""
Simple predictor interface using unified CostModel.
"""

from typing import Dict, Any, Optional, List
from gswarm.model import CostModel
from gswarm.utils import normalize_device


class Predictor:
    """
    Simple predictor class with unified interface for all model types.
    
    Example:
        predictor = Predictor()
        
        # Predict LLM inference time
        time = predictor.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello world"})
        
        # Predict diffusion model time
        time = predictor.predict("stable-diffusion", "node2:cuda:1", {"height": 512, "width": 512})
        
        # Update with actual time
        predictor.update("gpt-4", "node1:cuda:0", {"prompt": "Hello", "output": "Hi"}, actual_time=0.5)
    """
    
    def __init__(self, model_type: str = "auto"):
        """
        Initialize predictor.
        
        Args:
            model_type: Model type ("llm", "diffusion", or "auto" for automatic detection)
        """
        self.cost_model = CostModel(model_type=model_type)
    
    def predict(
        self, 
        model_name: str, 
        device: str, 
        inputs: Dict[str, Any],
        model_type: Optional[str] = None
    ) -> float:
        """
        Predict inference time.
        
        Args:
            model_name: Name of the model
            device: Device to run on (e.g., "node1:cuda:0" or legacy "cuda:0")
            inputs: Input features
            model_type: Optional model type override
            
        Returns:
            Estimated execution time in seconds
        """
        # Device normalization is handled by CostModel
        return self.cost_model.predict(model_name, device, inputs, model_type)
    
    def update(
        self,
        model_name: str,
        device: str,
        inputs: Dict[str, Any],
        actual_time: float,
        model_type: Optional[str] = None
    ) -> None:
        """
        Update model with actual execution time.
        
        Args:
            model_name: Name of the model
            device: Device used (e.g., "node1:cuda:0" or legacy "cuda:0")
            inputs: Input features used
            actual_time: Actual execution time in seconds
            model_type: Optional model type override
        """
        # Device normalization is handled by CostModel
        self.cost_model.update(model_name, device, inputs, actual_time, model_type)
    
    def batch_predict(
        self,
        requests: list[Dict[str, Any]]
    ) -> list[float]:
        """
        Predict inference time for multiple requests.
        
        Args:
            requests: List of request dictionaries, each containing:
                - model_name: str
                - device: str
                - inputs: Dict[str, Any]
                - model_type: Optional[str]
                
        Returns:
            List of estimated execution times
        """
        results = []
        for req in requests:
            time = self.predict(
                req["model_name"],
                req["device"],
                req["inputs"],
                req.get("model_type")
            )
            results.append(time)
        return results


# Global predictor instance for convenience
predictor = Predictor()