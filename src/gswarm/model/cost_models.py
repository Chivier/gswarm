"""
Cost models for predicting model inference times.
Integrated from gswarm-standalone-scheduler/standalone_model_server/
"""

import os
import random
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from gswarm.utils import normalize_device, get_device_key
from gswarm.utils.config import load_config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken not available - LLM token counting will be limited")


class LLMCostModel:
    """Cost model for predicting LLM inference time based on input/output tokens."""
    
    def __init__(self, train_data_path: Optional[str] = None, model_save_path: Optional[str] = None):
        """
        Initialize the LLMCostModel class.
        This class is responsible for loading training data, tokenizing it,
        training a predictor model, and making predictions based on input prompts.
        """
        self.config = load_config().predictor
        self.cost_model = self.initialize_model(train_data_path, model_save_path)
        self.new_data_buffer = []
        self.training_data_size = 0
        self.retrain_count = 0
        self.requests_since_last_train = 0

    def load_train_data(self, train_data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load training data from a specified path.
        If no path is provided, it returns an empty list.
        """
        if train_data_path is None:
            # Download default dataset
            default_dataset = "https://raw.githubusercontent.com/tloen/alpaca-lora/refs/heads/main/alpaca_data.json"
            try:
                import requests
                response = requests.get(default_dataset)
                data = response.json()
                self.training_data_size = len(data)
                return data
            except Exception as e:
                print(f"Failed to download default dataset: {e}")
                return []

        with open(train_data_path, "r") as file:
            data = json.load(file)
        self.training_data_size = len(data)
        return data

    def tokenize_dataset(self, train_data: List[Dict[str, Any]]) -> List[Dict[str, List[int]]]:
        """
        Tokenize the training data using tiktoken.
        """
        if not train_data:
            raise ValueError("Training data is empty. Please provide valid training data.")
        
        if not TIKTOKEN_AVAILABLE:
            print("WARNING: tiktoken not available, using character-based approximation")
            # Fallback: approximate tokens as words
            tokenized_data = []
            for item in train_data:
                tokenized_data.append({
                    "instruction": len(item.get("instruction", "").split()),
                    "input": len(item.get("input", "").split()),
                    "output": len(item.get("output", "").split()),
                })
            return tokenized_data

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

    def train_predictor(self, train_data: List[Dict[str, Any]]) -> LinearRegression:
        """
        Train a predictor model using the provided training data.
        """
        if not train_data:
            raise ValueError("Training data is empty. Please provide valid training data.")

        tokenized_data = self.tokenize_dataset(train_data)
        # Simulate training a linear regression model
        X = [[len(item["instruction"]) + len(item["input"]), item.get("gpu_utilization", 0), item.get("gpu_memory_cost", 0)] for item in tokenized_data]
        y = [len(item["output"]) for item in tokenized_data]
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
            try:
                model = joblib.load(model_save_path)
                # A way to get the training data size would be needed here.
                # For now, we'll assume it's a fixed number or load it from a separate file.
                self.training_data_size = 1000 # Placeholder
                return model
            except Exception as e:
                print(f"Failed to load model from {model_save_path}: {e}")

        try:
            train_data = self.load_train_data(train_data_path)
            if not train_data:
                raise ValueError("No training data found.")

            model = self.train_predictor(train_data)
            if model_save_path:
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

    def predict(self, prompt_length: int, gpu_utilization: float = 0, gpu_memory_cost: float = 0) -> int:
        """
        Predict the number of output tokens for a given input prompt using the trained model.
        """
        if not self.cost_model:
            raise RuntimeError("Model is not trained.")

        prediction = self.cost_model.predict([[prompt_length, gpu_utilization, gpu_memory_cost]])

        return int(prediction[0]) if prediction else 0

    def predict_str(self, prompt: str, gpu_utilization: float = 0, gpu_memory_cost: float = 0) -> int:
        """
        Predict the number of output tokens for a given input prompt string.
        """
        if not self.cost_model:
            raise RuntimeError("Model is not trained.")

        if TIKTOKEN_AVAILABLE:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
            input_tokens = tokenizer.encode(prompt)
            input_length = len(input_tokens)
        else:
            # Fallback: approximate tokens as words
            input_length = len(prompt.split())

        # Use the model to predict the number of output tokens
        prediction = self.cost_model.predict([[input_length, gpu_utilization, gpu_memory_cost]])

        return int(prediction[0]) if prediction else 0

    def _retrain(self):
        if not self.new_data_buffer:
            return
        
        print(f"Retraining LLM model with {len(self.new_data_buffer)} new samples.")
        self.cost_model = self.train_predictor(self.new_data_buffer)
        self.training_data_size += len(self.new_data_buffer)
        self.new_data_buffer = []
        self.retrain_count += 1
        self.requests_since_last_train = 0

    def update_model(self, new_data: List[Dict[str, Any]]):
        """
        Update the existing model with new training data.
        """
        if not self.cost_model:
            raise RuntimeError("Model is not trained.")

        if not new_data:
            raise ValueError("New data is empty. Please provide valid new data.")

        self.new_data_buffer.extend(new_data)
        self.requests_since_last_train += len(new_data)

        should_retrain = False
        if self.config.retrain_limit.policy == "max-try" and self.retrain_count >= self.config.retrain_limit.max_tries:
            return

        if self.config.update_strategy == "continuous":
            if len(self.new_data_buffer) >= self.config.continuous_update_trigger:
                should_retrain = True
        elif self.config.update_strategy == "incremental":
            if len(self.new_data_buffer) >= self.training_data_size:
                should_retrain = True
        
        if self.config.retrain_limit.policy == "interval" and self.requests_since_last_train >= self.config.retrain_limit.interval_requests:
            should_retrain = True

        if should_retrain:
            self._retrain()


class SDCostModel:
    """Cost model for predicting Stable Diffusion inference time based on image dimensions."""
    
    def __init__(self):
        self.config = load_config().predictor
        self.cost_model_mapping = {}
        self.new_data_buffer = {}
        self.training_data_size = {}
        self.retrain_count = {}
        self.requests_since_last_train = {}

    def initialize_model(self, model_name: str, device_name: str) -> Optional[LinearRegression]:
        """
        Initialize the model for predicting SD computing time.
        If a model save path is provided, load the model from that path.
        Otherwise, return a new LinearRegression model.
        """
        # Normalize device name for consistent storage
        normalized_device = get_device_key(device_name)
        predictor_name = f"{model_name.replace('/', '_')}_{normalized_device.replace(':', '_')}_predictor.pkl"

        home_path = os.path.expanduser("~")
        model_save_path = os.path.join(home_path, ".gswarm", predictor_name)

        # Check if the model is already loaded
        if predictor_name in self.cost_model_mapping:
            return self.cost_model_mapping[predictor_name]
        # If a model save path is provided, try to load the model from that path
        elif os.path.exists(model_save_path):
            try:
                # Load the model from the specified path
                model = joblib.load(model_save_path)
                self.training_data_size[predictor_name] = 100 # Placeholder
                return model
            except Exception as e:
                print(f"Error loading model from {model_save_path}: {e}")
        # If loading fails, create a new model
        else:
            print(f"No model found at {model_save_path}. A new model will be created.")
            # If no model is loaded, return a new LinearRegression instance
            self.cost_model_mapping[predictor_name] = self.create_model()
            self.training_data_size[predictor_name] = 4
        
        self.new_data_buffer[predictor_name] = []
        self.retrain_count[predictor_name] = 0
        self.requests_since_last_train[predictor_name] = 0
        return self.cost_model_mapping.get(predictor_name)

    def create_model(self) -> LinearRegression:
        """Create a new LinearRegression model."""
        model = LinearRegression()
        # Fit with some default data to avoid NotFittedError
        # Using height, width as features
        X = np.array([[128, 128, 0, 0], [256, 256, 0, 0], [512, 512, 0, 0], [1024, 1024, 0, 0]])
        y = np.array([0.5, 1.0, 2.5, 8.0])  # Approximate times
        model.fit(X, y)
        return model

    def _retrain(self, model_key: str):
        if not self.new_data_buffer.get(model_key):
            return
        
        print(f"Retraining SD model {model_key} with {len(self.new_data_buffer[model_key])} new samples.")
        
        X_new = np.array([item[0] for item in self.new_data_buffer[model_key]])
        y_new = np.array([item[1] for item in self.new_data_buffer[model_key]])

        self.cost_model_mapping[model_key].fit(X_new, y_new)
        self.training_data_size[model_key] += len(self.new_data_buffer[model_key])
        self.new_data_buffer[model_key] = []
        self.retrain_count[model_key] += 1
        self.requests_since_last_train[model_key] = 0

    def update_model(self, model_name: str, device_name: str, height: int, width: int, execution_time: float, gpu_utilization: float = 0, gpu_memory_cost: float = 0):
        """
        Update the model with new data.
        This function assumes that the model is a LinearRegression instance.
        """
        # Normalize device name for consistent storage
        normalized_device = get_device_key(device_name)
        model_key = f"{model_name.replace('/', '_')}_{normalized_device.replace(':', '_')}_predictor.pkl"
        if model_key not in self.cost_model_mapping:
            self.initialize_model(model_name, device_name)

        self.new_data_buffer.setdefault(model_key, []).append(([height, width, gpu_utilization, gpu_memory_cost], execution_time))
        self.requests_since_last_train[model_key] = self.requests_since_last_train.get(model_key, 0) + 1

        should_retrain = False
        if self.config.retrain_limit.policy == "max-try" and self.retrain_count.get(model_key, 0) >= self.config.retrain_limit.max_tries:
            return

        if self.config.update_strategy == "continuous":
            if len(self.new_data_buffer[model_key]) >= self.config.continuous_update_trigger:
                should_retrain = True
        elif self.config.update_strategy == "incremental":
            if len(self.new_data_buffer[model_key]) >= self.training_data_size.get(model_key, 0):
                should_retrain = True
        
        if self.config.retrain_limit.policy == "interval" and self.requests_since_last_train.get(model_key, 0) >= self.config.retrain_limit.interval_requests:
            should_retrain = True

        if should_retrain:
            self._retrain(model_key)


    def predict(self, model_name: str, device_name: str, height: int, width: int, gpu_utilization: float = 0, gpu_memory_cost: float = 0) -> float:
        """
        Predict the execution time using the model.
        This function assumes that the model is a LinearRegression instance.
        """
        # Reshape the input data
        X = np.array([[height, width, gpu_utilization, gpu_memory_cost]])

        # Normalize device name for consistent storage
        normalized_device = get_device_key(device_name)
        model_key = f"{model_name.replace('/', '_')}_{normalized_device.replace(':', '_')}_predictor.pkl"
        if model_key not in self.cost_model_mapping:
            # Initialize with default model if not exists
            self.cost_model_mapping[model_key] = self.initialize_model(model_name, device_name)
            if self.cost_model_mapping[model_key] is None:
                # Create default model with simple heuristic
                self.cost_model_mapping[model_key] = self.create_model()

        # Make the prediction
        return self.cost_model_mapping[model_key].predict(X)[0]


class CostModel:
    """
    Unified interface for all cost models.
    Provides a simple API for cost estimation regardless of model type.
    """
    
    def __init__(self, model_type: str = "auto"):
        """
        Initialize cost model.
        
        Args:
            model_type: Type of model ("llm", "diffusion", or "auto" for automatic detection)
        """
        self.model_type = model_type
        self._llm_model = None
        self._sd_model = None
    
    @property
    def llm_model(self) -> LLMCostModel:
        """Lazy initialization of LLM model."""
        if self._llm_model is None:
            self._llm_model = LLMCostModel()
        return self._llm_model
    
    @property
    def sd_model(self) -> SDCostModel:
        """Lazy initialization of SD model."""
        if self._sd_model is None:
            self._sd_model = SDCostModel()
        return self._sd_model
    
    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from model name."""
        name_lower = model_name.lower()
        if any(x in name_lower for x in ["stable-diffusion", "sd-", "flux", "diffusion"]):
            return "diffusion"
        elif any(x in name_lower for x in ["llama", "gpt", "mistral", "gemma", "qwen", "chat", "instruct"]):
            return "llm"
        else:
            return "llm"  # Default to LLM
    
    def predict(
        self, 
        model_name: str, 
        device: str, 
        inputs: Dict[str, Any],
        model_type: Optional[str] = None
    ) -> float:
        """
        Predict inference time for given inputs.
        
        Args:
            model_name: Name of the model
            device: Device to run on (e.g., "node1:cuda:0" or legacy "cuda:0")
            inputs: Input features for prediction
            model_type: Optional model type override
            
        Returns:
            Estimated execution time in seconds
        """
        # Normalize device name
        device = normalize_device(device)
        # Determine model type
        if model_type:
            actual_type = model_type
        elif self.model_type == "auto":
            actual_type = self._detect_model_type(model_name)
        else:
            actual_type = self.model_type
        
        gpu_utilization = inputs.get("gpu_utilization", 0)
        gpu_memory_cost = inputs.get("gpu_memory_cost", 0)

        # Prepare features based on model type
        if actual_type == "llm":
            if "prompt" in inputs:
                feature = {"prompt": inputs["prompt"]}
            elif "prompt_length" in inputs:
                feature = {"prompt_length": inputs["prompt_length"]}
            else:
                feature = {"prompt_length": inputs.get("max_tokens", 100)}
            
            feature["gpu_utilization"] = gpu_utilization
            feature["gpu_memory_cost"] = gpu_memory_cost
            result = self._predict_llm(model_name, device, [feature])
            
        elif actual_type == "diffusion":
            feature = {
                "height": inputs.get("height", 512),
                "width": inputs.get("width", 512),
                "gpu_utilization": gpu_utilization,
                "gpu_memory_cost": gpu_memory_cost,
            }
            result = self._predict_diffusion(model_name, device, [feature])
            
        else:
            # Unknown model type
            result = [random.uniform(1.0, 10.0)]
        
        return result[0] if result else 0.0
    
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
        # Normalize device name
        device = normalize_device(device)
        # Determine model type
        if model_type:
            actual_type = model_type
        elif self.model_type == "auto":
            actual_type = self._detect_model_type(model_name)
        else:
            actual_type = self.model_type
        
        gpu_utilization = inputs.get("gpu_utilization", 0)
        gpu_memory_cost = inputs.get("gpu_memory_cost", 0)

        # Prepare update data based on model type
        if actual_type == "llm":
            if "prompt" in inputs and "output" in inputs:
                feature = {
                    "prompt": inputs["prompt"],
                    "output": inputs["output"]
                }
            elif "prompt_length" in inputs and "output_length" in inputs:
                feature = {
                    "prompt_length": inputs["prompt_length"],
                    "output_length": inputs["output_length"]
                }
            else:
                return  # Cannot update without proper data
            
            feature["gpu_utilization"] = gpu_utilization
            feature["gpu_memory_cost"] = gpu_memory_cost
            self._update_llm(model_name, device, [feature])
            
        elif actual_type == "diffusion":
            feature = {
                "height": inputs.get("height", 512),
                "width": inputs.get("width", 512),
                "processing_time": actual_time,
                "gpu_utilization": gpu_utilization,
                "gpu_memory_cost": gpu_memory_cost,
            }
            self._update_diffusion(model_name, device, [feature])
    
    def _predict_llm(self, model_name: str, device: str, features: List[Dict[str, Any]]) -> List[float]:
        """Internal method for LLM prediction."""
        results = []
        for feature in features:
            gpu_utilization = feature.get("gpu_utilization", 0)
            gpu_memory_cost = feature.get("gpu_memory_cost", 0)
            if "prompt_length" in feature:
                results.append(self.llm_model.predict(feature["prompt_length"], gpu_utilization, gpu_memory_cost))
            elif "prompt" in feature:
                results.append(self.llm_model.predict_str(feature["prompt"], gpu_utilization, gpu_memory_cost))
            else:
                results.append(0.0)
        return results
    
    def _predict_diffusion(self, model_name: str, device: str, features: List[Dict[str, Any]]) -> List[float]:
        """Internal method for diffusion prediction."""
        results = []
        for feature in features:
            results.append(
                self.sd_model.predict(
                    model_name, device, 
                    feature["height"], feature["width"],
                    feature.get("gpu_utilization", 0),
                    feature.get("gpu_memory_cost", 0)
                )
            )
        return results
    
    def _update_llm(self, model_name: str, device: str, features: List[Dict[str, Any]]) -> None:
        """Internal method for LLM model update."""
        for feature in features:
            update_data = {
                "gpu_utilization": feature.get("gpu_utilization", 0),
                "gpu_memory_cost": feature.get("gpu_memory_cost", 0),
            }
            if "prompt" in feature and "output" in feature:
                update_data.update({
                    "instruction": feature["prompt"],
                    "input": "",
                    "output": feature["output"]
                })
                self.llm_model.update_model([update_data])
            elif "prompt_length" in feature and "output_length" in feature:
                # Create synthetic data
                update_data.update({
                    "instruction": "x" * feature["prompt_length"],
                    "input": "",
                    "output": "y" * feature["output_length"]
                })
                self.llm_model.update_model([update_data])
    
    def _update_diffusion(self, model_name: str, device: str, features: List[Dict[str, Any]]) -> None:
        """Internal method for diffusion model update."""
        for feature in features:
            self.sd_model.update_model(
                model_name, device,
                feature["height"], feature["width"],
                feature["processing_time"],
                feature.get("gpu_utilization", 0),
                feature.get("gpu_memory_cost", 0)
            )


# Global instances removed - use CostModel class instead for lazy initialization
# For backward compatibility, create functions that return instances
def get_sd_cost_model():
    """Get SD cost model instance (lazy initialization)."""
    if not hasattr(get_sd_cost_model, '_instance'):
        get_sd_cost_model._instance = SDCostModel()
    return get_sd_cost_model._instance

def get_llm_cost_model():
    """Get LLM cost model instance (lazy initialization)."""
    if not hasattr(get_llm_cost_model, '_instance'):
        get_llm_cost_model._instance = LLMCostModel()
    return get_llm_cost_model._instance

# Deprecated: Direct access to global instances
# Use get_sd_cost_model() or get_llm_cost_model() instead
sd_cost_model = None
llm_cost_model = None

# Global unified model
cost_model = CostModel()


def get_estimation_cost(model_type: str, model_name: str, device: str, data_features: List[Dict[str, Any]]) -> List[float]:
    """
    Get the estimation cost of a model for given data features.
    
    For LLM models:
    {"prompt_length": int} or {"prompt": str}
    
    For Diffusion models:
    {"height": int, "width": int}
    """
    # Normalize device name
    device = normalize_device(device)
    result = []
    for current_feature in data_features:
        gpu_utilization = current_feature.get("gpu_utilization", 0)
        gpu_memory_cost = current_feature.get("gpu_memory_cost", 0)
        if model_type == "llm":
            if "prompt_length" in current_feature:
                result.append(llm_cost_model.predict(current_feature["prompt_length"], gpu_utilization, gpu_memory_cost))
            elif "prompt" in current_feature:
                result.append(llm_cost_model.predict_str(current_feature["prompt"], gpu_utilization, gpu_memory_cost))
            else:
                raise ValueError("Invalid data features for LLM model prediction.")
        elif model_type == "diffusion":
            if "height" not in current_feature or "width" not in current_feature:
                raise ValueError("Diffusion models require 'height' and 'width' in data features")
            result.append(
                sd_cost_model.predict(model_name, device, current_feature["height"], current_feature["width"], gpu_utilization, gpu_memory_cost)
            )
        else:
            result.append(random.uniform(1.0, 100))  # Return a random cost for unknown models
    return result


def update_predictor(model_type: str, model_name: str, device: str, data_features: List[Dict[str, Any]]) -> None:
    """
    Update the predictor model with new data features.
    
    For LLM models:
    {"prompt_length": int, "output_length": int} or {"prompt": str, "output": str}
    
    For Diffusion models:
    {"height": int, "width": int, "processing_time": float}
    """
    # Normalize device name
    device = normalize_device(device)
    for current_feature in data_features:
        gpu_utilization = current_feature.get("gpu_utilization", 0)
        gpu_memory_cost = current_feature.get("gpu_memory_cost", 0)
        if model_type == "llm":
            update_data = {
                "gpu_utilization": gpu_utilization,
                "gpu_memory_cost": gpu_memory_cost,
            }
            if "prompt_length" not in current_feature and "prompt" in current_feature and "output" in current_feature:
                update_data.update({
                    "instruction": current_feature["prompt"],
                    "input": "",
                    "output": current_feature["output"]
                })
                llm_cost_model.update_model([update_data])
            elif "prompt_length" in current_feature and "output_length" in current_feature:
                # Create synthetic data for update
                update_data.update({
                    "instruction": "x" * current_feature["prompt_length"],
                    "input": "",
                    "output": "y" * current_feature["output_length"]
                })
                llm_cost_model.update_model([update_data])
            else:
                raise ValueError("Invalid data features for LLM model update.")
        elif model_type == "diffusion":
            if not all(k in current_feature for k in ["height", "width", "processing_time"]):
                raise ValueError("Diffusion models require 'height', 'width', and 'processing_time' for updates")
            sd_cost_model.update_model(
                model_name,
                device,
                current_feature["height"],
                current_feature["width"],
                current_feature["processing_time"],
                gpu_utilization,
                gpu_memory_cost
            )
        else:
            pass  # Ignore unknown model types
