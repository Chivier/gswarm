"""
Connection information management for gswarm.
Stores connection details in temporary files instead of config files.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from loguru import logger
import tempfile
import atexit
import getpass


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
    control_port: Optional[int] = None  # For client connections
    connection_type: str = "client"  # "client" or "host"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionInfo":
        return cls(**data)


class ConnectionManager:
    """Manages connection information in temporary files"""

    # Use a fixed location for connection info so all processes can find it
    # FIXME: Use a more robust location if needed (e.g., user-specific directory)
    CONNECTION_DIR = Path(tempfile.gettempdir()) / f"gswarm_connections_{getpass.getuser()}"

    def __init__(self, connection_type: Literal["client", "host"] = "client"):
        # Create the shared connection directory
        self.CONNECTION_DIR.mkdir(exist_ok=True)
        self.connection_type = connection_type

        # Use node-specific connection file with type prefix
        import platform

        node_name = platform.node().replace("/", "_").replace(":", "_")

        # Include connection type in filename to avoid conflicts when host and client are on same machine
        self.connection_file = self.CONNECTION_DIR / f"{connection_type}_{node_name}.json"

        # Only register cleanup if we're the process that created the connection
        # (checked when saving)
        self._should_cleanup = False

    def save_connection(self, info: ConnectionInfo) -> bool:
        """Save connection information to temp file"""
        try:
            info.pid = os.getpid()
            info.connection_type = self.connection_type
            with open(self.connection_file, "w") as f:
                json.dump(info.to_dict(), f, indent=2)
            logger.debug(f"Saved {self.connection_type} connection info to {self.connection_file}")

            # Mark for cleanup since we created it
            self._should_cleanup = True
            atexit.register(self.cleanup)

            return True
        except Exception as e:
            logger.error(f"Failed to save connection info: {e}")
            return False

    def load_connection(self) -> Optional[ConnectionInfo]:
        """Load connection information from temp file"""
        try:
            # Look for connection file with correct type
            if not self.connection_file.exists():
                # Try to find any connection file of the same type
                for conn_file in self.CONNECTION_DIR.glob(f"{self.connection_type}_*.json"):
                    self.connection_file = conn_file
                    break
                else:
                    return None

            with open(self.connection_file, "r") as f:
                data = json.load(f)

            # Verify connection type matches
            if data.get("connection_type", "client") != self.connection_type:
                logger.debug(
                    f"Connection type mismatch: expected {self.connection_type}, got {data.get('connection_type')}"
                )
                return None

            # Check if the process that created this is still running
            if "pid" in data and data["pid"]:
                try:
                    import psutil

                    if not psutil.pid_exists(data["pid"]):
                        logger.debug(f"Connection info from dead process (PID {data['pid']}), cleaning up")
                        self.cleanup()
                        return None
                except ImportError:
                    # If psutil not available, check with os.kill
                    try:
                        os.kill(data["pid"], 0)
                    except ProcessLookupError:
                        logger.debug(f"Connection info from dead process (PID {data['pid']}), cleaning up")
                        self.cleanup()
                        return None

            return ConnectionInfo.from_dict(data)
        except Exception as e:
            logger.debug(f"Failed to load connection info: {e}")
            return None

    def cleanup(self):
        """Remove connection info file"""
        try:
            if self.connection_file.exists():
                # Only cleanup if we're the process that created it OR if the process is dead
                should_cleanup = self._should_cleanup

                if not should_cleanup:
                    # Check if the creating process is dead
                    try:
                        info = self.load_connection()
                        if info and info.pid:
                            try:
                                os.kill(info.pid, 0)
                            except ProcessLookupError:
                                should_cleanup = True
                    except:
                        pass

                if should_cleanup:
                    self.connection_file.unlink()
                    logger.debug(f"Cleaned up {self.connection_type} connection info file")
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


# Global connection manager instance for backward compatibility (defaults to client)
connection_manager = ConnectionManager("client")

# Global connection manager instances for different connection types
_client_connection_manager = connection_manager  # Reuse the default instance
_host_connection_manager = None


def get_connection_manager(connection_type: Literal["client", "host"] = "client") -> ConnectionManager:
    """Get the appropriate connection manager based on type"""
    global _client_connection_manager, _host_connection_manager

    if connection_type == "client":
        return _client_connection_manager
    else:
        if _host_connection_manager is None:
            _host_connection_manager = ConnectionManager("host")
        return _host_connection_manager


def save_connection(
    host: str,
    profiler_grpc_port: int = 8090,
    profiler_http_port: int = 8091,
    model_api_port: int = 9010,
    node_id: Optional[str] = None,
    control_port: Optional[int] = None,
    is_host: bool = False,
) -> bool:
    """Save host connection information"""
    connection_type = "host" if is_host else "client"
    manager = get_connection_manager(connection_type)

    info = ConnectionInfo(
        host_address=host,
        profiler_grpc_port=profiler_grpc_port,
        profiler_http_port=profiler_http_port,
        model_api_port=model_api_port,
        connected_at=datetime.now().isoformat(),
        node_id=node_id,
        control_port=control_port,
        connection_type=connection_type,
    )
    return manager.save_connection(info)


def update_connection_info(key: str, value: Any, connection_type: Literal["client", "host"] = "client"):
    """Update a specific key in the connection information"""
    manager = get_connection_manager(connection_type)
    info = manager.load_connection()

    if info is None:
        logger.warning(f"No existing connection info found for {connection_type}")
        return False

    # Update the specified key
    setattr(info, key, value)

    # Save the updated connection info
    return manager.save_connection(info)


def get_connection_info(connection_type: Literal["client", "host"] = "client") -> Optional[ConnectionInfo]:
    """Get current connection information"""
    manager = get_connection_manager(connection_type)
    return manager.load_connection()


def get_connection_file(connection_type: Literal["client", "host"] = "client") -> str:
    """Get connection file path"""
    manager = get_connection_manager(connection_type)
    return str(manager.connection_file)


def clear_connection_info(connection_type: Literal["client", "host"] = "client"):
    """Clear connection information"""
    manager = get_connection_manager(connection_type)
    manager.cleanup()
