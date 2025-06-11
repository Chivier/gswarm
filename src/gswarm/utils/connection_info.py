"""
Connection information management for gswarm.
Stores connection details in temporary files instead of config files.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from loguru import logger
import tempfile
import atexit


@dataclass
class ConnectionInfo:
    """Connection information for gswarm services"""

    host_address: str
    profiler_grpc_port: int
    profiler_http_port: int
    model_api_port: int
    connected_at: str
    node_id: Optional[str] = None
    pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionInfo":
        return cls(**data)


class ConnectionManager:
    """Manages connection information in temporary files"""

    def __init__(self):
        uuid6 = str(uuid.uuid4())[:6]
        self.temp_dir = Path(tempfile.gettempdir()) / f"gswarm_{uuid6}"
        self.temp_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.connection_file = self.temp_dir / f"connection_info_{timestamp}.json"

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def save_connection(self, info: ConnectionInfo) -> bool:
        """Save connection information to temp file"""
        try:
            info.pid = os.getpid()
            with open(self.connection_file, "w") as f:
                json.dump(info.to_dict(), f, indent=2)
            logger.debug(f"Saved connection info to {self.connection_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save connection info: {e}")
            return False

    def load_connection(self) -> Optional[ConnectionInfo]:
        """Load connection information from temp file"""
        try:
            if not self.connection_file.exists():
                return None

            with open(self.connection_file, "r") as f:
                data = json.load(f)

            # Check if the process that created this is still running
            if "pid" in data and data["pid"]:
                try:
                    import psutil

                    if not psutil.pid_exists(data["pid"]):
                        logger.debug("Connection info from dead process, ignoring")
                        self.cleanup()
                        return None
                except ImportError:
                    # If psutil not available, just trust the file
                    pass

            return ConnectionInfo.from_dict(data)
        except Exception as e:
            logger.debug(f"Failed to load connection info: {e}")
            return None

    def cleanup(self):
        """Remove connection info file"""
        try:
            if self.connection_file.exists():
                self.connection_file.unlink()
                logger.debug(f"Cleaned up connection info file")
        except Exception as e:
            logger.debug(f"Failed to cleanup connection info: {e}")

    def get_model_api_url(self) -> str:
        """Get model API URL from connection info or default"""
        info = self.load_connection()
        if info:
            return f"http://{info.host_address}:{info.model_api_port}"
        return "http://localhost:9010"  # Default

    def get_profiler_grpc_address(self) -> str:
        """Get profiler gRPC address from connection info or default"""
        info = self.load_connection()
        if info:
            return f"{info.host_address}:{info.profiler_grpc_port}"
        return "localhost:8090"  # Default

    def get_profiler_http_url(self) -> str:
        """Get profiler HTTP URL from connection info or default"""
        info = self.load_connection()
        if info:
            return f"http://{info.host_address}:{info.profiler_http_port}"
        return "http://localhost:8091"  # Default


# Global connection manager instance
connection_manager = ConnectionManager()


def save_host_connection(
    host: str,
    profiler_grpc_port: int = 8090,
    profiler_http_port: int = 8091,
    model_api_port: int = 9010,
    node_id: Optional[str] = None,
) -> bool:
    """Save host connection information"""
    info = ConnectionInfo(
        host_address=host,
        profiler_grpc_port=profiler_grpc_port,
        profiler_http_port=profiler_http_port,
        model_api_port=model_api_port,
        connected_at=datetime.now().isoformat(),
        node_id=node_id,
    )
    return connection_manager.save_connection(info)


def get_connection_info() -> Optional[ConnectionInfo]:
    """Get current connection information"""
    return connection_manager.load_connection()


def get_connection_file() -> str:
    """Get connection file path"""
    return connection_manager.connection_file


def clear_connection_info():
    """Clear connection information"""
    connection_manager.cleanup()
