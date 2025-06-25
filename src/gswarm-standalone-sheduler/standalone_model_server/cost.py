import random

def get_estimation_cost(model_name: str, device: str, *data_features: str) -> float:
    """
    Get the estimation cost of a model for given data features.
    """
    return random.uniform(0.0, 100.0)