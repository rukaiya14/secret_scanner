"""
Configuration management for the ML-Enhanced Secret Scanner.

This module handles loading, validation, and management of configuration
from YAML or JSON files with safe defaults.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ml_scanner.exceptions import ConfigLoadError, ConfigValidationError
from ml_scanner.logger import get_logger

logger = get_logger(__name__)


# Default configuration values
DEFAULT_CONFIG = {
    "ml_scanner": {
        "confidence_thresholds": {
            "blocking": 0.90,
            "warning": 0.85,
        },
        "timeouts": {
            "inference": 5,
            "explanation": 2,
            "commit_scan": 10,
        },
        "detection_categories": {
            "API_KEY": True,
            "PASSWORD": True,
            "PII": True,
            "TOKEN": True,
            "CERTIFICATE": True,
            "OTHER": True,
        },
        "batch_size": 16,
        "explainer_backend": "SHAP",  # SHAP or LIME
        "device": "auto",  # auto, cpu, cuda
    },
    "model_registry": {
        "storage_path": "./model_registry",
        "max_versions": 5,
        "cache_enabled": True,
    },
    "training": {
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_sequence_length": 512,
        "class_balance_ratio": 3.0,
        "augmentation_enabled": True,
        "min_precision": 0.90,
        "min_recall": 0.85,
    },
    "monitoring": {
        "metrics_enabled": True,
        "prometheus_port": 9090,
        "daily_reports": True,
    },
    "alerting": {
        "slack_webhook_url": None,
        "retry_attempts": 3,
        "retry_backoff": [1, 2, 4],
    },
}


class Config:
    """
    Configuration manager for the ML scanner.
    
    Handles loading configuration from YAML/JSON files, validation,
    and providing safe defaults for missing values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON).
                        If None, uses default configuration.
        """
        # Deep copy to avoid modifying the default config
        import copy
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        
        if config_path:
            self.load_config(config_path)
        
        self.validate_config()
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigLoadError: If file cannot be loaded
        """
        try:
            path = Path(config_path)
            
            if not path.exists():
                raise ConfigLoadError(
                    f"Configuration file not found: {config_path}",
                    context={"path": config_path}
                )
            
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif path.suffix == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ConfigLoadError(
                        f"Unsupported configuration file format: {path.suffix}",
                        context={"path": config_path, "suffix": path.suffix}
                    )
            
            # Merge loaded config with defaults (loaded config takes precedence)
            self._merge_config(self.config, loaded_config)
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigLoadError(
                f"Failed to parse configuration file: {str(e)}",
                context={"path": config_path, "error": str(e)}
            )
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to load configuration: {str(e)}",
                context={"path": config_path, "error": str(e)}
            )
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """
        Recursively merge override config into base config.
        
        Args:
            base: Base configuration dictionary (modified in place)
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            # Validate confidence thresholds
            blocking = self.config["ml_scanner"]["confidence_thresholds"]["blocking"]
            warning = self.config["ml_scanner"]["confidence_thresholds"]["warning"]
            
            if not (0.0 <= blocking <= 1.0):
                raise ConfigValidationError(
                    f"Blocking threshold must be between 0.0 and 1.0, got {blocking}",
                    context={"blocking_threshold": blocking}
                )
            
            if not (0.0 <= warning <= 1.0):
                raise ConfigValidationError(
                    f"Warning threshold must be between 0.0 and 1.0, got {warning}",
                    context={"warning_threshold": warning}
                )
            
            if warning > blocking:
                logger.warning(
                    f"Warning threshold ({warning}) is higher than blocking threshold ({blocking}). "
                    "This may cause unexpected behavior."
                )
            
            # Validate timeouts
            for timeout_name, timeout_value in self.config["ml_scanner"]["timeouts"].items():
                if timeout_value <= 0:
                    raise ConfigValidationError(
                        f"Timeout '{timeout_name}' must be positive, got {timeout_value}",
                        context={"timeout_name": timeout_name, "timeout_value": timeout_value}
                    )
            
            # Validate explainer backend
            explainer = self.config["ml_scanner"]["explainer_backend"]
            if explainer not in ["SHAP", "LIME"]:
                raise ConfigValidationError(
                    f"Explainer backend must be 'SHAP' or 'LIME', got '{explainer}'",
                    context={"explainer_backend": explainer}
                )
            
            # Validate device
            device = self.config["ml_scanner"]["device"]
            if device not in ["auto", "cpu", "cuda"]:
                raise ConfigValidationError(
                    f"Device must be 'auto', 'cpu', or 'cuda', got '{device}'",
                    context={"device": device}
                )
            
            # Validate batch size
            batch_size = self.config["ml_scanner"]["batch_size"]
            if batch_size <= 0:
                raise ConfigValidationError(
                    f"Batch size must be positive, got {batch_size}",
                    context={"batch_size": batch_size}
                )
            
            # Validate training parameters
            if self.config["training"]["min_precision"] < 0 or self.config["training"]["min_precision"] > 1:
                raise ConfigValidationError(
                    f"Minimum precision must be between 0 and 1",
                    context={"min_precision": self.config["training"]["min_precision"]}
                )
            
            if self.config["training"]["min_recall"] < 0 or self.config["training"]["min_recall"] > 1:
                raise ConfigValidationError(
                    f"Minimum recall must be between 0 and 1",
                    context={"min_recall": self.config["training"]["min_recall"]}
                )
            
            logger.info("Configuration validation successful")
            
        except KeyError as e:
            raise ConfigValidationError(
                f"Missing required configuration key: {str(e)}",
                context={"missing_key": str(e)}
            )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "ml_scanner.batch_size")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, output_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save configuration file (YAML or JSON)
        """
        path = Path(output_path)
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(self.config, f, indent=2)
            else:
                raise ConfigLoadError(
                    f"Unsupported output format: {path.suffix}",
                    context={"path": output_path}
                )
        
        logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Config instance
    """
    return Config(config_path)
