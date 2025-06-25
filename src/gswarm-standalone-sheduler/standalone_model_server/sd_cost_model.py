from sklearn.linear_model import LinearRegression
from typing import Optional

import os


class SDCostModel:
    def __init__(self):
        self.cost_model_mapping = {}

    def initialize_model(self, model_name, device_name) -> Optional[LinearRegression]:
        """
        Initialize the model for predicting SD computing time.
        If a model save path is provided, load the model from that path.
        Otherwise, return a new LinearRegression model.
        """
        predictor_name = f"{model_name.replace('/', '_')}_{device_name}_predictor.pkl"

        home_path = os.path.expanduser("~")
        model_save_path = os.path.join(home_path, ".gswarm", predictor_name)

        # Check if the model is already loaded
        if predictor_name in self.cost_model_mapping:
            return self.cost_model_mapping[predictor_name]
        # If a model save path is provided, try to load the model from that path
        elif os.path.exists(model_save_path):
            import joblib

            try:
                # Load the model from the specified path
                return joblib.load(model_save_path)
            except Exception as e:
                print(f"Error loading model from {model_save_path}: {e}")
        # If loading fails, create a new model
        else:
            print(f"No model found at {model_save_path}. A new model will be created.")
            # If no model is loaded, return a new LinearRegression instance
            self.cost_model_mapping[predictor_name] = self.create_model()
        return self.cost_model_mapping.get(predictor_name)

    def create_model(self):
        # TODO: Implement the logic to create a new model.
        # For now, we will return a new LinearRegression instance.
        return LinearRegression()

    def update_model(self, model_name, device_name, height, width, execution_time):
        """
        Update the model with new data.
        This function assumes that the model is a LinearRegression instance.
        """
        import numpy as np

        # Reshape the input data
        X = np.array([[height, width]])
        y = np.array([execution_time])

        model_key = f"{model_name.replace('/', '_')}_{device_name}_predictor.pkl"
        if model_key not in self.cost_model_mapping:
            self.cost_model_mapping[model_key] = self.create_model()

        self.cost_model_mapping[model_key].fit(X, y)

    def predict(self, model_name, device_name, height, width):
        """
        Predict the execution time using the model.
        This function assumes that the model is a LinearRegression instance.
        """
        import numpy as np

        # Reshape the input data
        X = np.array([[height, width]])

        # Make the prediction
        return self.cost_model_mapping[f"{model_name.replace('/', '_')}_{device_name}_predictor.pkl"].predict(X)[0]


def initialize_predictor(model_name: str, device_name: str):
    """
    Initialize the predictor for the specified model.
    This function is a placeholder and should be implemented based on the specific model requirements.
    """
    predictor_name = f"{model_name.replace('/', '_')}_{device_name}_predictor.pkl"

    home_path = os.path.expanduser("~")
    model_save_path = os.path.join(home_path, ".gswarm", predictor_name)

    cost_model = SDCostModel(model_save_path=model_save_path)
    return cost_model
