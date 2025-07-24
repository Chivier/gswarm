"""
Base Model Service Interface for GSwarm

This module provides a flexible base class for implementing custom model services.
Users can extend this class to create their own model serving implementations
with custom start, stop, and inference logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import signal
import sys
from datetime import datetime
import os


@dataclass
class ServiceConfig:
    """Configuration for a model service"""
    name: str
    model_path: str
    host: str = "0.0.0.0"
    port: Optional[int] = None  # Will be assigned by model manager
    device: str = "cpu"
    max_batch_size: int = 1
    timeout: float = 30.0
    extra_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}


class InferenceRequest(BaseModel):
    """Base inference request model"""
    inputs: Union[Dict[str, Any], List[Any], str]
    parameters: Optional[Dict[str, Any]] = None


class InferenceResponse(BaseModel):
    """Base inference response model"""
    outputs: Union[Dict[str, Any], List[Any], str]
    metadata: Optional[Dict[str, Any]] = None


class ModelService(ABC):
    """
    Abstract base class for model services.
    
    Users should inherit from this class and implement the abstract methods
    to create custom model serving implementations.
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app = FastAPI(title=f"{config.name} Model Service")
        self._setup_routes()
        self._server = None
        self._is_running = False
        self._start_time = None
        
    def _setup_routes(self):
        """Set up FastAPI routes"""
        
        @self.app.get("/")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self._is_running else "starting",
                "service": self.config.name,
                "uptime": self._get_uptime(),
                "device": self.config.device,
                "port": self.config.port
            }
        
        @self.app.get("/stop")
        async def stop_service():
            """Stop the service gracefully"""
            logger.info(f"Stopping service {self.config.name}")
            asyncio.create_task(self._shutdown())
            return {"message": "Service is stopping"}
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """Main inference endpoint"""
            if not self._is_running:
                raise HTTPException(status_code=503, detail="Service is not ready")
            
            try:
                result = await self.inference(
                    request.inputs,
                    request.parameters or {}
                )
                return InferenceResponse(outputs=result)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def get_status():
            """Get detailed service status"""
            return await self.get_status()
    
    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the model service.
        This method should load the model and prepare for inference.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the model service and clean up resources.
        """
        pass
    
    @abstractmethod
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        """
        Perform inference on the given inputs.
        
        Args:
            inputs: Input data for inference
            parameters: Additional parameters for inference
            
        Returns:
            Inference results
        """
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the service.
        Can be overridden to provide custom status information.
        """
        return {
            "name": self.config.name,
            "status": "running" if self._is_running else "stopped",
            "uptime": self._get_uptime(),
            "device": self.config.device,
            "port": self.config.port,
            "model_path": self.config.model_path,
            "max_batch_size": self.config.max_batch_size
        }
    
    def _get_uptime(self) -> Optional[float]:
        """Get service uptime in seconds"""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None
    
    async def _shutdown(self):
        """Internal shutdown handler"""
        await self.stop()
        self._is_running = False
        if self._server:
            self._server.should_exit = True
    
    def run(self):
        """Run the service"""
        async def startup():
            """Startup event handler"""
            logger.info(f"Starting {self.config.name} service on port {self.config.port}")
            await self.start()
            self._is_running = True
            self._start_time = datetime.now()
            logger.info(f"{self.config.name} service started successfully")
        
        async def shutdown():
            """Shutdown event handler"""
            logger.info(f"Shutting down {self.config.name} service")
            await self.stop()
            self._is_running = False
        
        # Add event handlers
        self.app.add_event_handler("startup", startup)
        self.app.add_event_handler("shutdown", shutdown)
        
        # Handle signals
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(self._shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the server
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        
        try:
            asyncio.run(self._server.serve())
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        finally:
            logger.info(f"{self.config.name} service stopped")


class BatchModelService(ModelService):
    """
    Extended base class that supports batch inference.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self._setup_batch_routes()
    
    def _setup_batch_routes(self):
        """Set up batch inference routes"""
        
        @self.app.post("/inference_batch")
        async def inference_batch(requests: List[InferenceRequest]):
            """Batch inference endpoint"""
            if not self._is_running:
                raise HTTPException(status_code=503, detail="Service is not ready")
            
            if len(requests) > self.config.max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size {len(requests)} exceeds maximum {self.config.max_batch_size}"
                )
            
            try:
                results = await self.inference_batch(
                    [r.inputs for r in requests],
                    [r.parameters or {} for r in requests]
                )
                return [InferenceResponse(outputs=result) for result in results]
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    @abstractmethod
    async def inference_batch(
        self,
        inputs_list: List[Any],
        parameters_list: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Perform batch inference on multiple inputs.
        
        Args:
            inputs_list: List of input data
            parameters_list: List of parameters for each input
            
        Returns:
            List of inference results
        """
        pass