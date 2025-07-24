from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from gswarm.model import CostModel
from gswarm.utils import normalize_device


class PredictorOutput(BaseModel):
    model_name: str
    device_name: str
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def predict_inference(model_name: str, device_name: str, inputs: Dict[str, Any]) -> PredictorOutput:
    """
    Predict inference time for a given model and device.

    Args:
        model_name: Name of the model
        device_name: Device to run on (e.g., 'node1:cuda:0' or legacy 'cuda:0')
        inputs: Input data for the model

    Returns:
        PredictorOutput: Output containing prediction or error message.
    """
    try:
        # Normalize device name
        normalized_device = normalize_device(device_name)
        
        # Use unified CostModel interface
        cost_model = CostModel(model_type="auto")
        
        # Get estimation
        estimated_time = cost_model.predict(
            model_name=model_name,
            device=normalized_device,
            inputs=inputs
        )
        
        return PredictorOutput(
            model_name=model_name,
            device_name=normalized_device,
            prediction={"inference_time": estimated_time},
        )
        
    except Exception as e:
        return PredictorOutput(
            model_name=model_name,
            device_name=normalize_device(device_name),
            error=str(e)
        )


def update_model_predictor(model_name: str, device_name: str, inputs: Dict[str, Any], actual_time: float) -> bool:
    """
    Update the predictor model with actual execution time.
    
    Args:
        model_name: Name of the model
        device_name: Device name (e.g., 'node1:cuda:0' or legacy 'cuda:0')
        inputs: Input data for the model
        actual_time: Actual execution time in seconds
        
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Normalize device name
        normalized_device = normalize_device(device_name)
        
        # Use unified CostModel interface
        cost_model = CostModel(model_type="auto")
        
        # Update model
        cost_model.update(
            model_name=model_name,
            device=normalized_device,
            inputs=inputs,
            actual_time=actual_time
        )
        
        return True
        
    except Exception as e:
        print(f"Error updating predictor: {e}")
        return False
