import typer
import json
import sys
import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from gswarm.predictor.feature_factory import feature_factory, ModelType, ModelFeatures
from gswarm.predictor.predicting import predict_inference

app = typer.Typer()

@app.command()
def predict(
    model_name: str,
    device_name: str,
    model_type: ModelType,
    features: Dict[str, Any],
):
    """
    Predict inference time for a given model and device.
    
    Args:
        model_name (str): Name of the model.
        device_name (str): Name of the device.
        model_type (ModelType): Type of the model.
        features (Dict[str, Any]): Features for the model.
    
    Returns:
        None
    """
    try:
        feature = feature_factory(model_name, device_name, model_type, features)
        prediction = predict_inference(feature)
        typer.echo(json.dumps(prediction.model_dump(), indent=2))
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)