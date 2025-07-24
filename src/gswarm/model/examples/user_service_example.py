#!/usr/bin/env python3
"""
Example: Creating a Custom Model Service

This example shows how users can create their own model service
by extending the base ModelService class.
"""

import numpy as np
from typing import Any, Dict
from gswarm.model import ModelService, ServiceConfig
from loguru import logger


class MyCustomOCRService(ModelService):
    """
    Example custom OCR service similar to the OnnxOCR-Ray pattern.
    
    This demonstrates how to implement:
    - Model initialization in start()
    - Resource cleanup in stop()
    - Custom inference logic
    - Error handling
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.model = None
        self.preprocessor = None
        
    async def start(self) -> None:
        """Initialize the OCR model and preprocessor"""
        logger.info(f"Loading OCR model from {self.config.model_path}")
        
        # Simulate loading an OCR model
        # In real implementation, you would load your actual model here
        import time
        time.sleep(1)  # Simulate loading time
        
        self.model = {"type": "ocr", "version": "1.0"}
        self.preprocessor = {"resize": (640, 640)}
        
        logger.info("OCR model loaded successfully")
        
    async def stop(self) -> None:
        """Clean up model resources"""
        logger.info("Stopping OCR service")
        self.model = None
        self.preprocessor = None
        
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        """
        Perform OCR inference on input image.
        
        Args:
            inputs: Base64 encoded image or image path
            parameters: Additional parameters like 'language', 'confidence_threshold'
            
        Returns:
            Dict with detected text and bounding boxes
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Extract parameters
        language = parameters.get("language", "en")
        conf_threshold = parameters.get("confidence_threshold", 0.5)
        
        # Simulate OCR processing
        # In real implementation, you would:
        # 1. Decode base64 image if needed
        # 2. Preprocess the image
        # 3. Run OCR model
        # 4. Post-process results
        
        # Simulated results
        import random
        num_detections = random.randint(1, 5)
        
        detections = []
        for i in range(num_detections):
            detection = {
                "text": f"Text_{i}",
                "confidence": random.uniform(0.7, 0.99),
                "bbox": [
                    random.randint(0, 100),
                    random.randint(0, 100),
                    random.randint(100, 200),
                    random.randint(100, 200)
                ]
            }
            detections.append(detection)
        
        # Filter by confidence
        detections = [d for d in detections if d["confidence"] >= conf_threshold]
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "language": language,
            "processing_time_ms": random.uniform(50, 200)
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Provide custom status information"""
        status = await super().get_status()
        
        # Add OCR-specific status
        status.update({
            "model_type": "OCR",
            "supported_languages": ["en", "zh", "ja", "ko"],
            "model_loaded": self.model is not None,
            "preprocessor_loaded": self.preprocessor is not None,
        })
        
        return status


def main():
    """
    Example of running the service standalone
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom OCR Model Service")
    parser.add_argument("--model", required=True, help="Path to OCR model")
    parser.add_argument("--port", type=int, default=8080, help="Service port")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create service configuration
    config = ServiceConfig(
        name="ocr-service",
        model_path=args.model,
        port=args.port,
        device=args.device,
        max_batch_size=10,
        extra_args={
            "model_type": "onnx",
            "input_size": (640, 640)
        }
    )
    
    # Create and run service
    service = MyCustomOCRService(config)
    service.run()


if __name__ == "__main__":
    main()