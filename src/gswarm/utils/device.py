"""
Device notation utilities for GSwarm.

Supports device formats:
- client:device:0 (e.g., "node1:cuda:0", "worker1:cuda:1")
- Legacy format: cuda:0 (will be converted to default client)
"""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Parsed device information."""
    client: str
    device_type: str  # cuda, cpu, etc.
    device_id: int
    
    @property
    def full_name(self) -> str:
        """Get full device name in format client:device:id"""
        return f"{self.client}:{self.device_type}:{self.device_id}"
    
    @property
    def local_name(self) -> str:
        """Get local device name (without client)"""
        return f"{self.device_type}:{self.device_id}"
    
    @property
    def legacy_name(self) -> str:
        """Get legacy device name for backward compatibility"""
        return self.local_name


def parse_device(device_str: str, default_client: str = "localhost") -> DeviceInfo:
    """
    Parse device string into components.
    
    Args:
        device_str: Device string in format "client:device:id" or "device:id"
        default_client: Default client name if not specified
        
    Returns:
        DeviceInfo with parsed components
        
    Examples:
        >>> parse_device("node1:cuda:0")
        DeviceInfo(client='node1', device_type='cuda', device_id=0)
        
        >>> parse_device("cuda:0")  # Legacy format
        DeviceInfo(client='localhost', device_type='cuda', device_id=0)
        
        >>> parse_device("worker2:cuda:3")
        DeviceInfo(client='worker2', device_type='cuda', device_id=3)
    """
    parts = device_str.split(":")
    
    if len(parts) == 3:
        # New format: client:device:id
        client = parts[0]
        device_type = parts[1]
        try:
            device_id = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid device ID in '{device_str}': {parts[2]}")
            
    elif len(parts) == 2:
        # Legacy format: device:id
        client = default_client
        device_type = parts[0]
        try:
            device_id = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid device ID in '{device_str}': {parts[1]}")
            
    else:
        raise ValueError(
            f"Invalid device format '{device_str}'. "
            f"Expected 'client:device:id' or 'device:id'"
        )
    
    return DeviceInfo(client=client, device_type=device_type, device_id=device_id)


def format_device(client: str, device_type: str, device_id: int) -> str:
    """
    Format device components into standard notation.
    
    Args:
        client: Client/node name
        device_type: Device type (cuda, cpu, etc.)
        device_id: Device ID number
        
    Returns:
        Formatted device string
    """
    return f"{client}:{device_type}:{device_id}"


def normalize_device(device_str: str, default_client: str = "localhost") -> str:
    """
    Normalize device string to standard format.
    
    Args:
        device_str: Device string in any supported format
        default_client: Default client if not specified
        
    Returns:
        Normalized device string in format "client:device:id"
    """
    device_info = parse_device(device_str, default_client)
    return device_info.full_name


def is_same_device(device1: str, device2: str, ignore_client: bool = False) -> bool:
    """
    Check if two devices are the same.
    
    Args:
        device1: First device string
        device2: Second device string
        ignore_client: If True, only compare device type and ID
        
    Returns:
        True if devices are the same
    """
    d1 = parse_device(device1)
    d2 = parse_device(device2)
    
    if ignore_client:
        return d1.device_type == d2.device_type and d1.device_id == d2.device_id
    else:
        return d1.client == d2.client and d1.device_type == d2.device_type and d1.device_id == d2.device_id


def get_device_key(device_str: str) -> str:
    """
    Get a normalized key for device-based dictionaries.
    
    This ensures consistent keys regardless of input format.
    """
    return normalize_device(device_str)


# Backward compatibility helpers
def convert_legacy_device(device_str: str, client: Optional[str] = None) -> str:
    """
    Convert legacy device format to new format.
    
    Args:
        device_str: Device string (may be legacy format)
        client: Client name to use (defaults to hostname if not specified)
        
    Returns:
        Device string in new format
    """
    if client is None:
        import socket
        client = socket.gethostname()
    
    return normalize_device(device_str, client)