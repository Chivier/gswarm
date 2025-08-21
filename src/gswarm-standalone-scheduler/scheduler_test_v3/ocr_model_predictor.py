import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import hashlib
import random


class TimeDataset(Dataset):
    """PyTorch dataset for time prediction data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class TimePredictorDNN(nn.Module):
    """Simple DNN for time prediction"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(TimePredictorDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class OCRModelPredictor:
    """Lightweight predictor for OCR model execution time estimation"""
    
    def __init__(self, 
                 data_dir: str = "./predictor_data",
                 retrain_threshold: int = 100,
                 max_logs: int = 10000):
        """
        Initialize the predictor
        
        Args:
            data_dir: Directory to store logs and models
            retrain_threshold: Number of new records before retraining
            max_logs: Maximum number of logs to store (FIFO)
        """
        self.data_dir = data_dir
        self.retrain_threshold = retrain_threshold
        self.max_logs = max_logs
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths
        self.log_file = os.path.join(self.data_dir, "training_log.json")
        self.model_file = os.path.join(self.data_dir, "model.pkl")
        
        # Initialize components
        self.model = None
        self.model_name_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Load existing data
        self.records = self._load_records()
        self.new_records_count = 0
        
        # Load existing model if available
        self._load_model()
    
    def _load_records(self) -> List[Dict]:
        """Load existing records from file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                records = json.load(f)
                # Ensure we don't exceed max_logs
                if len(records) > self.max_logs:
                    records = records[-self.max_logs:]
                return records
        return []
    
    def _save_records(self):
        """Save records to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.records, f, indent=2)
    
    def _load_model(self):
        """Load existing model if available"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                
                # Recreate model
                self.model = TimePredictorDNN(saved_data['input_dim'])
                self.model.load_state_dict(saved_data['model_state'])
                self.model.eval()
                
                # Load encoders and scaler
                self.model_name_encoder = saved_data['model_name_encoder']
                self.device_encoder = saved_data['device_encoder']
                self.feature_scaler = saved_data['feature_scaler']
                
                print(f"Loaded existing model trained on {len(self.records)} records")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def _save_model(self, input_dim: int):
        """Save model and preprocessing components"""
        saved_data = {
            'input_dim': input_dim,
            'model_state': self.model.state_dict(),
            'model_name_encoder': self.model_name_encoder,
            'device_encoder': self.device_encoder,
            'feature_scaler': self.feature_scaler
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(saved_data, f)
    
    def _extract_features(self, params: Dict) -> List[float]:
        """Extract numerical features from params dictionary"""
        features = []
        
        # Common OCR model parameters with defaults
        features.append(params.get('batch_size', 1))
        features.append(params.get('image_width', 640))
        features.append(params.get('image_height', 480))
        features.append(params.get('num_classes', 10))
        features.append(params.get('sequence_length', 100))
        features.append(params.get('hidden_size', 256))
        features.append(params.get('num_layers', 2))
        features.append(params.get('dropout', 0.1))
        
        # Add hash of params for capturing other variations
        params_str = json.dumps(params, sort_keys=True)
        params_hash = int(hashlib.md5(params_str.encode()).hexdigest()[:8], 16)
        features.append(params_hash % 1000000)
        
        return features
    
    def predict(self, model_name: str, params: Dict, device: str) -> float:
        """
        Predict execution time for given inputs
        
        Args:
            model_name: Name of the OCR model
            params: Model parameters dictionary
            device: Device type (e.g., 'cpu', 'cuda:0')
            
        Returns:
            Estimated finish time in seconds
        """
        # If no model is trained, return random estimate
        if self.model is None:
            random_time = random.uniform(10, 300)
            print(f"No trained model, returning random estimate: {random_time:.2f}s")
            return random_time
        
        try:
            # Prepare features
            features = []
            
            # Handle categorical features
            if model_name in self.model_name_encoder.classes_:
                model_encoded = self.model_name_encoder.transform([model_name])[0]
            else:
                model_encoded = len(self.model_name_encoder.classes_)
            features.append(model_encoded)
            
            if device in self.device_encoder.classes_:
                device_encoded = self.device_encoder.transform([device])[0]
            else:
                device_encoded = len(self.device_encoder.classes_)
            features.append(device_encoded)
            
            # Add numerical features
            features.extend(self._extract_features(params))
            
            # Scale and predict
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features_array)
            
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                prediction = self.model(features_tensor).item()
            
            # Ensure positive prediction
            return max(prediction, 1.0)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return random.uniform(10, 300)
    
    def add_record(self, input_data: Dict, time_cost: float):
        """
        Add a new training record
        
        Args:
            input_data: Dictionary with keys 'model_name', 'params', 'device'
            time_cost: Actual execution time in seconds
        """
        # Create record
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': input_data['model_name'],
            'params': input_data['params'],
            'device': input_data['device'],
            'time_cost': time_cost
        }
        
        # Add to records
        self.records.append(record)
        
        # If we exceed max_logs, remove oldest records (FIFO)
        if len(self.records) > self.max_logs:
            # Keep only the most recent max_logs records
            self.records = self.records[-self.max_logs:]
            print(f"Reached max_logs limit ({self.max_logs}), removed oldest records")
        
        self.new_records_count += 1
        
        # Save to file
        self._save_records()
        
        print(f"Added record #{len(self.records)}. New records: {self.new_records_count}")
        
        # Check if we should retrain
        if self.new_records_count >= self.retrain_threshold:
            print(f"Reached {self.retrain_threshold} new records. Retraining model...")
            self.train()
            self.new_records_count = 0
    
    def train(self, epochs: int = 100, batch_size: int = 32):
        """Train the model on all available records"""
        if len(self.records) < 10:
            print(f"Not enough data to train ({len(self.records)} records). Need at least 10.")
            return
        
        print(f"Training on {len(self.records)} records...")
        
        # Prepare data
        model_names = []
        devices = []
        all_features = []
        targets = []
        
        for record in self.records:
            model_names.append(record['model_name'])
            devices.append(record['device'])
            all_features.append(self._extract_features(record['params']))
            targets.append(record['time_cost'])
        
        # Fit encoders
        self.model_name_encoder.fit(model_names)
        self.device_encoder.fit(devices)
        
        # Encode categorical features
        model_encoded = self.model_name_encoder.transform(model_names)
        device_encoded = self.device_encoder.transform(devices)
        
        # Combine features
        features = np.column_stack([model_encoded, device_encoded, all_features])
        targets = np.array(targets)
        
        # Scale features
        self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = TimeDataset(X_train, y_train)
        val_dataset = TimeDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = features.shape[1]
        self.model = TimePredictorDNN(input_dim)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_features).squeeze()
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    predictions = self.model(batch_features).squeeze()
                    loss = criterion(predictions, batch_targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(input_dim)
            
            if (epoch + 1) % 20 == 0:
                avg_train_loss = train_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'total_records': len(self.records),
            'new_records': self.new_records_count,
            'model_trained': self.model is not None,
            'retrain_threshold': self.retrain_threshold,
            'max_logs': self.max_logs,
            'storage_usage': f"{len(self.records)}/{self.max_logs}"
        }
    
    def predict_task_duration(self, model_name: str, input_data: Dict) -> float:
        """
        Predict execution time for a single task
        
        Args:
            model_name: Name of the OCR model
            input_data: Input data dictionary containing model parameters
            
        Returns:
            Estimated execution time in seconds
        """
        # Extract device from input_data if present, otherwise default to 'cpu'
        device = input_data.get('device', 'cpu')
        params = input_data.get('params', input_data)
        
        return self.predict(model_name, params, device)
    
    def get_model_performance_stats(self, model_name: str) -> Dict:
        """
        Get historical performance statistics for a specific model
        
        Args:
            model_name: Name of the OCR model
            
        Returns:
            Dictionary containing performance statistics
        """
        model_records = [r for r in self.records if r['model_name'] == model_name]
        
        if not model_records:
            return {
                'model_name': model_name,
                'total_runs': 0,
                'average_time': 0,
                'min_time': 0,
                'max_time': 0,
                'std_time': 0
            }
        
        times = [r['time_cost'] for r in model_records]
        
        return {
            'model_name': model_name,
            'total_runs': len(model_records),
            'average_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def update_performance_history(self, task_id: str, actual_duration: float, input_data: Dict) -> None:
        """
        Update model performance history to improve prediction accuracy
        
        Args:
            task_id: Unique task identifier
            actual_duration: Actual execution time in seconds
            input_data: Dictionary with keys 'model_name', 'params', 'device'
        """
        # Add record with task_id for tracking
        record = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'model_name': input_data['model_name'],
            'params': input_data.get('params', {}),
            'device': input_data.get('device', 'cpu'),
            'time_cost': actual_duration
        }
        
        self.records.append(record)
        
        if len(self.records) > self.max_logs:
            self.records = self.records[-self.max_logs:]
        
        self.new_records_count += 1
        self._save_records()
        
        print(f"Updated performance for task {task_id}. New records: {self.new_records_count}")
        
        if self.new_records_count >= self.retrain_threshold:
            print(f"Reached {self.retrain_threshold} new records. Retraining model...")
            self.train()
            self.new_records_count = 0


# Example usage
if __name__ == "__main__":
    # Initialize predictor with max 5000 logs
    predictor = OCRModelPredictor(
        retrain_threshold=50,
        max_logs=5000
    )
    
    # Example input
    input_data = {
        'model_name': 'CRNN',
        'params': {
            'batch_size': 32,
            'image_width': 640,
            'image_height': 64,
            'num_classes': 37,
            'sequence_length': 50,
            'hidden_size': 256,
            'num_layers': 2
        },
        'device': 'cuda:0'
    }
    
    # Get prediction
    estimated_time = predictor.predict(
        input_data['model_name'], 
        input_data['params'], 
        input_data['device']
    )
    print(f"Estimated time: {estimated_time:.2f} seconds")
    
    # After execution, add actual time
    actual_time = 45.3
    predictor.add_record(input_data, actual_time)
    
    # Check statistics
    print(f"Statistics: {predictor.get_statistics()}")
