"""
Utility functions for gswarm
"""

from .cache import (
    get_cache_dir,
    get_model_cache_dir,
    clean_history,
    model_storage,
    scan_all_models,
    save_model_to_disk,
    load_safetensors_to_dram,
    scan_gswarm_models,
    scan_huggingface_models,
)
from .config import (
    load_config,
    save_config,
    get_config_path,
    get_model_cache_dir as get_config_model_cache_dir,
    get_dram_cache_dir,
    get_huggingface_cache_dir,
    GSwarmConfig,
    HostConfig,
    ClientConfig,
)
from .connection_info import (
    ConnectionInfo,
    ConnectionManager,
    connection_manager,
    save_connection,
    get_connection_info,
    clear_connection_info,
)

__all__ = [
    # Cache functions
    "get_cache_dir",
    "get_model_cache_dir",
    "clean_history",
    "model_storage",
    "scan_all_models",
    "save_model_to_disk",
    "load_safetensors_to_dram",
    "scan_gswarm_models",
    "scan_huggingface_models",
    # Configuration functions
    "load_config",
    "save_config",
    "get_config_path",
    "get_config_model_cache_dir",
    "get_dram_cache_dir",
    "get_huggingface_cache_dir",
    "GSwarmConfig",
    "HostConfig",
    "ClientConfig",
    # Connection info functions
    "ConnectionInfo",
    "ConnectionManager",
    "connection_manager",
    "save_connection",
    "get_connection_info",
    "clear_connection_info",
]
