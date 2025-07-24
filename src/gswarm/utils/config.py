"""
Simplified configuration management for gswarm.
"""

import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

# Global variable to store custom config path
_custom_config_path: Optional[Path] = None


@dataclass
class HostConfig:
    """Host/Head node configuration"""

    huggingface_cache_dir: str = "~/.cache/huggingface"
    model_cache_dir: str = "~/.cache/gswarm/models"
    model_manager_port: int = 8101
    gswarm_grpc_port: int = 8091
    gswarm_http_port: int = 8090


@dataclass
class ClientConfig:
    """Client node configuration"""

    host_url: str = "0.0.0.0:8091"
    dram_size: int = 16  # GB
    model_cache_dir: str = "~/.cache/gswarm/models"
    node_id: str = "node1"


@dataclass
class PredictorRetrainLimitConfig:
    """Configuration for predictor retraining limits."""
    policy: str = "unlimited"  # unlimited, interval, max-try
    interval_requests: int = 1000
    max_tries: int = 10

@dataclass
class PredictorConfig:
    """Configuration for the performance predictor."""
    update_strategy: str = "continuous"  # continuous, incremental
    continuous_update_trigger: int = 50
    retrain_limit: PredictorRetrainLimitConfig = field(default_factory=PredictorRetrainLimitConfig)


@dataclass
class GSwarmConfig:
    """Combined gswarm configuration"""

    host: HostConfig = field(default_factory=HostConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)


def set_config_path(config_path: Optional[str]) -> None:
    """Set the path to the configuration file"""
    global _custom_config_path
    if config_path:
        _custom_config_path = Path(config_path)
        logger.info(f"Using custom configuration file: {_custom_config_path}")
    else:
        _custom_config_path = None


def get_config_path() -> Path:
    """Get path to the configuration file"""
    global _custom_config_path
    if _custom_config_path:
        return _custom_config_path
    return Path.home() / ".gswarm.conf"


def load_config() -> GSwarmConfig:
    """Load configuration from YAML file or create default"""
    config_path = get_config_path()

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            # Load host config
            host_data = data.get("host", {})
            host_config = HostConfig(
                huggingface_cache_dir=host_data.get("huggingface_cache_dir", "~/.cache/huggingface"),
                model_cache_dir=host_data.get("model_cache_dir", "~/.cache/gswarm/models"),
                model_manager_port=host_data.get("model_manager_port", 8101),
                gswarm_grpc_port=host_data.get("gswarm_grpc_port", 8091),
                gswarm_http_port=host_data.get("gswarm_http_port", 8090),
            )

            # Load client config
            client_data = data.get("client", {})
            client_config = ClientConfig(
                host_url=client_data.get("host_url", "0.0.0.0:8091"),
                dram_size=client_data.get("dram_size", 16),
                model_cache_dir=client_data.get("model_cache_dir", "~/.cache/gswarm/models"),
                node_id=client_data.get("node_id", "node1"),
            )

            # Load predictor config
            predictor_data = data.get("predictor", {})
            retrain_limit_data = predictor_data.get("retrain_limit", {})
            retrain_limit_config = PredictorRetrainLimitConfig(
                policy=retrain_limit_data.get("policy", "unlimited"),
                interval_requests=retrain_limit_data.get("interval_requests", 1000),
                max_tries=retrain_limit_data.get("max_tries", 10),
            )
            predictor_config = PredictorConfig(
                update_strategy=predictor_data.get("update_strategy", "continuous"),
                continuous_update_trigger=predictor_data.get("continuous_update_trigger", 50),
                retrain_limit=retrain_limit_config,
            )

            config = GSwarmConfig(host=host_config, client=client_config, predictor=predictor_config)
            logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {e}")
            logger.info("Using default configuration")

    # Return default config without saving if file doesn't exist
    config = GSwarmConfig()
    logger.info("Using default configuration (no config file found)")
    return config


def save_config(config: GSwarmConfig) -> bool:
    """Save configuration to YAML file"""
    config_path = get_config_path()

    try:
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "host": {
                "huggingface_cache_dir": config.host.huggingface_cache_dir,
                "model_cache_dir": config.host.model_cache_dir,
                "model_manager_port": config.host.model_manager_port,
                "gswarm_grpc_port": config.host.gswarm_grpc_port,
                "gswarm_http_port": config.host.gswarm_http_port,
            },
            "client": {
                "host_url": config.client.host_url,
                "dram_size": config.client.dram_size,
                "model_cache_dir": config.client.model_cache_dir,
                "node_id": config.client.node_id,
            },
            "predictor": {
                "update_strategy": config.predictor.update_strategy,
                "continuous_update_trigger": config.predictor.continuous_update_trigger,
                "retrain_limit": {
                    "policy": config.predictor.retrain_limit.policy,
                    "interval_requests": config.predictor.retrain_limit.interval_requests,
                    "max_tries": config.predictor.retrain_limit.max_tries,
                }
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def get_model_cache_dir(config: Optional[GSwarmConfig] = None) -> Path:
    """Get the model cache directory path"""
    if config is None:
        config = load_config()

    path = Path(config.host.model_cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dram_cache_dir() -> Path:
    """Get DRAM cache directory"""
    path = Path("/dev/shm/gswarm_models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_huggingface_cache_dir(config: Optional[GSwarmConfig] = None) -> Path:
    """Get HuggingFace cache directory for scanning existing models"""
    if config is None:
        config = load_config()

    path = Path(config.host.huggingface_cache_dir).expanduser()
    return path
