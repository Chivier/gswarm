import random
import os

from typing import Optional

import tiktoken
from sklearn.linear_model import LinearRegression


class LLMCostModel:
    def __init__(self, train_data_path: Optional[str] = None, model_save_path: Optional[str] = None):
        """
        Initialize the LLMCostModel class.
        This class is responsible for loading training data, tokenizing it,
        training a predictor model, and making predictions based on input prompts.
        """
        self.cost_model = self.initialize_model(train_data_path, model_save_path)

    def load_train_data(self, train_data_path: Optional[str] = None):
        """
        Load training data from a specified path.
        If no path is provided, it returns an empty list.
        """

        if train_data_path is None:
            # Download default dataset
            default_dataset = "https://raw.githubusercontent.com/tloen/alpaca-lora/refs/heads/main/alpaca_data.json"
            import requests
            import json

            response = requests.get(default_dataset)
            data = response.json()
            return data

        with open(train_data_path, "r") as file:
            data = json.load(file)
        return data

    def tokenize_dataset(self, train_data):
        """
        Tokenize the training data using tiktoken.
        """
        if not train_data:
            raise ValueError("Training data is empty. Please provide valid training data.")

        tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Tokenize each instruction in the training data
        tokenized_data = [
            {
                "instruction": tokenizer.encode(item["instruction"]) if "instruction" in item else [],
                "input": tokenizer.encode(item["input"]) if "input" in item else [],
                "output": tokenizer.encode(item["output"]) if "output" in item else [],
            }
            for item in train_data
        ]

        return tokenized_data

    def train_predictor(self, train_data):
        """
        Train a predictor model using the provided training data.
        This is a placeholder function that simulates training.
        """
        if not train_data:
            raise ValueError("Training data is empty. Please provide valid training data.")

        train_data = self.tokenize_dataset(train_data)
        # Simulate training a linear regression model
        X = [[len(item["instruction"]) + len(item["input"])] for item in train_data]
        y = [len(item["output"]) for item in train_data]
        model = LinearRegression()
        model.fit(X, y)
        return model

    def initialize_model(
        self, train_data_path: Optional[str] = None, model_save_path: Optional[str] = None
    ) -> Optional[LinearRegression]:
        """
        Initialize and train the model using the provided training data path.
        If no path is provided, it uses a default dataset.
        """
        if model_save_path is None:
            home_path = os.path.expanduser("~")
            model_save_path = os.path.join(home_path, ".gswarm", "llm_cost_model.pkl")
            print(f"Save to default path: {model_save_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Load the model from the specified path if it exists
        if os.path.exists(model_save_path):
            import joblib

            return joblib.load(model_save_path)

        try:
            train_data = self.load_train_data(train_data_path)
            if not train_data:
                raise ValueError("No training data found.")

            model = self.train_predictor(train_data)
            if model_save_path:
                import joblib

                joblib.dump(model, model_save_path)
            print(
                f"Model initialized and saved to {model_save_path}"
                if model_save_path
                else "Model initialized without saving."
            )
            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None

    def predict(self, input_prompt: str) -> int:
        """
        Predict the number of output tokens for a given input prompt using the trained model.
        """
        if not self.cost_model:
            raise RuntimeError("Model is not trained.")

        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = tokenizer.encode(input_prompt)

        # Use the model to predict the number of output tokens
        input_length = len(input_tokens)
        prediction = self.cost_model.predict([[input_length]])

        return int(prediction[0]) if prediction else 0

    def update_model(self, new_data):
        """
        Update the existing model with new training data.
        This is a placeholder function that simulates updating the model.
        """
        if not self.cost_model:
            raise RuntimeError("Model is not trained.")

        if not new_data:
            raise ValueError("New data is empty. Please provide valid new data.")

        # Simulate updating the model with new data
        new_data = self.tokenize_dataset(new_data)
        X_new = [[len(item["instruction"]) + len(item["input"])] for item in new_data]
        y_new = [len(item["output"]) for item in new_data]

        self.cost_model.fit(X_new, y_new)
