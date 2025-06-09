"""
Configuration management for gswarm.
Handles loading and saving of YAML configuration files with separate host and client configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class HostConfig:
    """Host/Head node configuration"""
    host: str = "0.0.0.0"
    port: int = 8100
    model_port: int = 8101
    model_cache_dir: str = "~/.cache/gswarm/models"  # Fixed path
    log_level: str = "INFO"
    max_concurrent_downloads: int = 3
    auto_discover_models: bool = True
    cleanup_on_shutdown: bool = True
    huggingface_cache_dir: str = "~/.cache/huggingface"  # For scanning existing models
    storage_devices: Dict[str, Any] = field(default_factory=lambda: {
        "disk": {"enabled": True, "path": "~/.cache/gswarm/models"},
        "dram": {"enabled": True, "path": "/dev/shm/gswarm_models", "max_size_gb": 64}
    })
    # Model loading settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True


@dataclass 
class ClientConfig:
    """Client node configuration"""
    host_url: str = "http://localhost:8100"
    node_id: Optional[str] = None
    model_cache_dir: str = "~/.cache/gswarm/models"  # Fixed path
    heartbeat_interval: int = 30
    auto_register: bool = True
    auto_discover_models: bool = True
    log_level: str = "INFO"
    dram_cache_size: int = 16  # GB
    gpu_memory_fraction: float = 0.9
    # Model serving settings
    default_gpu_memory_utilization: float = 0.90
    enable_tensor_parallel: bool = True
    max_concurrent_requests: int = 256


@dataclass
class GSwarmConfig:
    """Combined gswarm configuration"""
    host: HostConfig = field(default_factory=HostConfig)
    client: ClientConfig = field(default_factory=ClientConfig)


def get_config_path() -> Path:
    """Get path to the configuration file"""
    config_path = Path.home() / ".gswarm.conf"
    return config_path


def load_config() -> GSwarmConfig:
    """Load configuration from YAML file or create default"""
    config_path = get_config_path()
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            # Load host config
            host_data = data.get('host', {})
            host_config = HostConfig(
                host=host_data.get('host', "0.0.0.0"),
                port=host_data.get('port', 8100),
                model_port=host_data.get('model_port', 8101),
                model_cache_dir=host_data.get('model_cache_dir', "~/.cache/gswarm/models"),
                log_level=host_data.get('log_level', "INFO"),
                max_concurrent_downloads=host_data.get('max_concurrent_downloads', 3),
                auto_discover_models=host_data.get('auto_discover_models', True),
                cleanup_on_shutdown=host_data.get('cleanup_on_shutdown', True),
                huggingface_cache_dir=host_data.get('huggingface_cache_dir', "~/.cache/huggingface"),
                storage_devices=host_data.get('storage_devices', {
                    "disk": {"enabled": True, "path": "~/.cache/gswarm/models"},
                    "dram": {"enabled": True, "path": "/dev/shm/gswarm_models", "max_size_gb": 64}
                }),
                load_in_8bit=host_data.get('load_in_8bit', False),
                load_in_4bit=host_data.get('load_in_4bit', False),
                trust_remote_code=host_data.get('trust_remote_code', True)
            )
            
            # Load client config
            client_data = data.get('client', {})
            client_config = ClientConfig(
                host_url=client_data.get('host_url', "http://localhost:8100"),
                node_id=client_data.get('node_id'),
                model_cache_dir=client_data.get('model_cache_dir', "~/.cache/gswarm/models"),
                heartbeat_interval=client_data.get('heartbeat_interval', 30),
                auto_register=client_data.get('auto_register', True),
                auto_discover_models=client_data.get('auto_discover_models', True),
                log_level=client_data.get('log_level', "INFO"),
                dram_cache_size=client_data.get('dram_cache_size', 16),
                gpu_memory_fraction=client_data.get('gpu_memory_fraction', 0.9),
                default_gpu_memory_utilization=client_data.get('default_gpu_memory_utilization', 0.90),
                enable_tensor_parallel=client_data.get('enable_tensor_parallel', True),
                max_concurrent_requests=client_data.get('max_concurrent_requests', 256)
            )
            
            config = GSwarmConfig(host=host_config, client=client_config)
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {e}")
            logger.info("Using default configuration")
    
    # Create default config
    config = GSwarmConfig()
    
    # Save default config for future use
    save_config(config)
    
    return config


def save_config(config: GSwarmConfig) -> bool:
    """Save configuration to YAML file"""
    config_path = get_config_path()
    
    try:
        # Convert to dict for YAML serialization
        data = {
            'host': {
                'host': config.host.host,
                'port': config.host.port,
                'model_port': config.host.model_port,
                'model_cache_dir': config.host.model_cache_dir,
                'log_level': config.host.log_level,
                'max_concurrent_downloads': config.host.max_concurrent_downloads,
                'auto_discover_models': config.host.auto_discover_models,
                'cleanup_on_shutdown': config.host.cleanup_on_shutdown,
                'huggingface_cache_dir': config.host.huggingface_cache_dir,
                'storage_devices': config.host.storage_devices,
                'load_in_8bit': config.host.load_in_8bit,
                'load_in_4bit': config.host.load_in_4bit,
                'trust_remote_code': config.host.trust_remote_code
            },
            'client': {
                'host_url': config.client.host_url,
                'node_id': config.client.node_id,
                'model_cache_dir': config.client.model_cache_dir,
                'heartbeat_interval': config.client.heartbeat_interval,
                'auto_register': config.client.auto_register,
                'auto_discover_models': config.client.auto_discover_models,
                'log_level': config.client.log_level,
                'dram_cache_size': config.client.dram_cache_size,
                'gpu_memory_fraction': config.client.gpu_memory_fraction,
                'default_gpu_memory_utilization': config.client.default_gpu_memory_utilization,
                'enable_tensor_parallel': config.client.enable_tensor_parallel,
                'max_concurrent_requests': config.client.max_concurrent_requests
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def get_model_cache_dir(config: Optional[GSwarmConfig] = None) -> Path:
    """Get the model cache directory path (fixed to ~/.cache/gswarm/models)"""
    if config is None:
        config = load_config()
    
    # Always use the fixed path
    cache_dir = config.host.model_cache_dir
    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dram_cache_dir() -> Path:
    """Get DRAM cache directory"""
    config = load_config()
    dram_path = config.host.storage_devices.get("dram", {}).get("path", "/dev/shm/gswarm_models")
    path = Path(dram_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_huggingface_cache_dir(config: Optional[GSwarmConfig] = None) -> Path:
    """Get HuggingFace cache directory for scanning existing models"""
    if config is None:
        config = load_config()
    
    hf_cache_dir = config.host.huggingface_cache_dir
    path = Path(hf_cache_dir).expanduser()
    return path 