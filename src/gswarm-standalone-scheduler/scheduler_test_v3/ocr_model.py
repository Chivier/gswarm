# API:

# localhost/model_name/start 
# - input: device_id, port(optional, if port is not given, use random available port between 10000-20000)
# - return: model_id (uuid), port

# localhost/model_name/stop {}
# - input: model_id
# - return status code (success/failed)

# localhost/model_name/inference {}
# - input: model_id, request data
# - output: ….

import requests
from .ocr_model_predictor import *

def start_model(url="localhost", model_name="detection", device_id="cuda:0", port=None):
    if port is None:
        port = random.randint(10000, 20000)
    response = requests.post(f"{url}/{model_name}/start", json={"device_id": device_id, "port": port})
    return response.json()

def stop_model(url="localhost", model_id=None):
    if model_id is None:
        raise ValueError("model_id is required")
    response = requests.post(f"{url}/{model_id}/stop", json={})
    return response.status_code

def inference(url="localhost", model_id=None, request_data=None):
    if model_id is None:
        raise ValueError("model_id is required")
    if request_data is None:
        raise ValueError("request_data is required")
    response = requests.post(f"{url}/{model_id}/inference", json={"request_data": request_data})
    return response.json()


class ModelInstance:
    def __init__(self, model_name, device_id, port=None):
        self.model_name = model_name
        self.device_id = device_id
        self.port = port
        self.model_id = None
        self.status = "idle"  # idle, busy, stopped
        self.current_task_id = None
    
    def start(self, url="localhost"):
        """Start the model instance"""
        response = start_model(url, self.model_name, self.device_id, self.port)
        self.model_id = response.get('model_id')
        self.port = response.get('port')
        self.status = "idle"
        return response
    
    def stop(self, url="localhost"):
        """Stop the model instance"""
        if self.model_id:
            status = stop_model(url, self.model_id)
            self.status = "stopped"
            return status
        return None
    
    def inference(self, request_data, url="localhost"):
        """Run inference on the model"""
        if not self.model_id:
            raise ValueError("Model not started")
        
        self.status = "busy"
        try:
            result = inference(url, self.model_id, request_data)
            return result
        finally:
            self.status = "idle"
            self.current_task_id = None
    
    def is_available(self):
        """Check if model instance is available for new tasks"""
        return self.status == "idle" and self.model_id is not None