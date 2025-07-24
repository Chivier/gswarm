"""
Model Service Manager

Manages the lifecycle of model services, including port allocation,
service registration, and process management.
"""

import asyncio
import subprocess
import sys
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import socket
from loguru import logger
import psutil
from datetime import datetime
import random

from .base_service import ModelService, ServiceConfig


@dataclass
class ServiceInfo:
    """Information about a running service"""
    name: str
    service_type: str
    config: ServiceConfig
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    status: str = "stopped"  # stopped, starting, running, stopping, failed
    error_message: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "service_type": self.service_type,
            "config": {
                "name": self.config.name,
                "model_path": self.config.model_path,
                "host": self.config.host,
                "port": self.config.port,
                "device": self.config.device,
                "max_batch_size": self.config.max_batch_size,
                "timeout": self.config.timeout,
                "extra_args": self.config.extra_args
            },
            "pid": self.pid,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "status": self.status,
            "error_message": self.error_message
        }


class ServiceManager:
    """
    Manages model services lifecycle and port allocation.
    """
    
    # Default port range for model services
    DEFAULT_PORT_RANGE = (8000, 9000)
    
    def __init__(self, port_range: tuple = None):
        self.port_range = port_range or self.DEFAULT_PORT_RANGE
        self.services: Dict[str, ServiceInfo] = {}
        self.allocated_ports: set = set()
        self._load_service_state()
    
    def _find_free_port(self) -> int:
        """Find a free port in the configured range"""
        min_port, max_port = self.port_range
        
        # Try random ports first for better distribution
        for _ in range(100):
            port = random.randint(min_port, max_port)
            if port not in self.allocated_ports and self._is_port_free(port):
                return port
        
        # Fall back to sequential search
        for port in range(min_port, max_port):
            if port not in self.allocated_ports and self._is_port_free(port):
                return port
        
        raise RuntimeError(f"No free ports available in range {self.port_range}")
    
    def _is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
    
    def _allocate_port(self, preferred_port: Optional[int] = None) -> int:
        """Allocate a port for a service"""
        if preferred_port:
            if preferred_port in self.allocated_ports:
                raise ValueError(f"Port {preferred_port} is already allocated")
            if not self._is_port_free(preferred_port):
                raise ValueError(f"Port {preferred_port} is not available")
            port = preferred_port
        else:
            port = self._find_free_port()
        
        self.allocated_ports.add(port)
        return port
    
    def _release_port(self, port: int):
        """Release an allocated port"""
        self.allocated_ports.discard(port)
    
    def register_service(
        self,
        name: str,
        service_type: str,
        config: ServiceConfig
    ) -> ServiceInfo:
        """Register a new service"""
        if name in self.services:
            raise ValueError(f"Service {name} already exists")
        
        # Allocate port if not specified
        if config.port is None:
            config.port = self._allocate_port()
        else:
            self._allocate_port(config.port)
        
        service_info = ServiceInfo(
            name=name,
            service_type=service_type,
            config=config
        )
        
        self.services[name] = service_info
        self._save_service_state()
        
        logger.info(f"Registered service {name} on port {config.port}")
        return service_info
    
    async def start_service(self, name: str) -> ServiceInfo:
        """Start a registered service"""
        if name not in self.services:
            raise ValueError(f"Service {name} not found")
        
        service_info = self.services[name]
        
        if service_info.status == "running":
            logger.warning(f"Service {name} is already running")
            return service_info
        
        try:
            service_info.status = "starting"
            self._save_service_state()
            
            # Prepare command to run the service
            cmd = [
                sys.executable,
                "-m", f"gswarm.model.service_runner",
                "--name", name,
                "--service-type", service_info.service_type,
                "--config", json.dumps(service_info.config.__dict__)
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            service_info.process = process
            service_info.pid = process.pid
            service_info.start_time = datetime.now()
            service_info.status = "running"
            service_info.error_message = None
            
            self._save_service_state()
            
            logger.info(f"Started service {name} with PID {process.pid}")
            return service_info
            
        except Exception as e:
            service_info.status = "failed"
            service_info.error_message = str(e)
            self._save_service_state()
            logger.error(f"Failed to start service {name}: {e}")
            raise
    
    async def stop_service(self, name: str, force: bool = False) -> bool:
        """Stop a running service"""
        if name not in self.services:
            raise ValueError(f"Service {name} not found")
        
        service_info = self.services[name]
        
        if service_info.status != "running":
            logger.warning(f"Service {name} is not running")
            return False
        
        try:
            service_info.status = "stopping"
            self._save_service_state()
            
            if service_info.process and service_info.process.poll() is None:
                if force:
                    service_info.process.kill()
                else:
                    service_info.process.terminate()
                
                # Wait for process to stop
                try:
                    service_info.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Service {name} did not stop gracefully, killing...")
                    service_info.process.kill()
                    service_info.process.wait()
            
            service_info.status = "stopped"
            service_info.process = None
            service_info.pid = None
            
            self._save_service_state()
            
            logger.info(f"Stopped service {name}")
            return True
            
        except Exception as e:
            service_info.status = "failed"
            service_info.error_message = str(e)
            self._save_service_state()
            logger.error(f"Failed to stop service {name}: {e}")
            raise
    
    def unregister_service(self, name: str):
        """Unregister a service"""
        if name not in self.services:
            raise ValueError(f"Service {name} not found")
        
        service_info = self.services[name]
        
        if service_info.status == "running":
            raise ValueError(f"Cannot unregister running service {name}")
        
        # Release port
        if service_info.config.port:
            self._release_port(service_info.config.port)
        
        del self.services[name]
        self._save_service_state()
        
        logger.info(f"Unregistered service {name}")
    
    def get_service_info(self, name: str) -> Optional[ServiceInfo]:
        """Get information about a service"""
        return self.services.get(name)
    
    def list_services(self) -> List[ServiceInfo]:
        """List all registered services"""
        return list(self.services.values())
    
    def get_service_status(self, name: str) -> Dict[str, Any]:
        """Get detailed status of a service"""
        if name not in self.services:
            raise ValueError(f"Service {name} not found")
        
        service_info = self.services[name]
        status = service_info.to_dict()
        
        # Check if process is actually running
        if service_info.pid:
            try:
                process = psutil.Process(service_info.pid)
                if process.is_running():
                    status["cpu_percent"] = process.cpu_percent()
                    status["memory_info"] = process.memory_info()._asdict()
                else:
                    service_info.status = "stopped"
                    service_info.pid = None
                    self._save_service_state()
            except psutil.NoSuchProcess:
                service_info.status = "stopped"
                service_info.pid = None
                self._save_service_state()
        
        return status
    
    def _get_state_file(self) -> Path:
        """Get the path to the service state file"""
        state_dir = Path.home() / ".gswarm" / "services"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "service_state.json"
    
    def _save_service_state(self):
        """Save service state to disk"""
        state_file = self._get_state_file()
        state = {
            "services": {
                name: info.to_dict()
                for name, info in self.services.items()
            },
            "allocated_ports": list(self.allocated_ports)
        }
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_service_state(self):
        """Load service state from disk"""
        state_file = self._get_state_file()
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            # Restore allocated ports
            self.allocated_ports = set(state.get("allocated_ports", []))
            
            # Restore services (but mark them as stopped since processes don't persist)
            for name, info_dict in state.get("services", {}).items():
                config_dict = info_dict["config"]
                config = ServiceConfig(**config_dict)
                
                service_info = ServiceInfo(
                    name=name,
                    service_type=info_dict["service_type"],
                    config=config,
                    status="stopped"  # Processes don't persist across restarts
                )
                
                self.services[name] = service_info
                
        except Exception as e:
            logger.error(f"Failed to load service state: {e}")


# Global service manager instance
_service_manager = None


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager