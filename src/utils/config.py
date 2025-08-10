"""
Configuration utilities for MOVAL experiments.

This module provides configuration management for different experiment
parameters and settings.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration class for experiments."""
    
    # Dataset configuration
    dataset_name: str
    data_path: str
    model_path: str
    
    # Experiment parameters
    random_seed: int = 42
    num_folds: int = 5
    batch_size: int = 32
    
    # Model parameters
    confidence_threshold: float = 0.5
    ensemble_size: int = 5
    
    # Output configuration
    output_dir: str = "results"
    save_models: bool = True
    save_predictions: bool = True
    
    # Evaluation parameters
    metrics: list = None
    confidence_bins: int = 10
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1_score']


class ConfigManager:
    """Manager for experiment configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = None
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        suffix = self.config_path.suffix.lower()
        
        if suffix == '.yaml' or suffix == '.yml':
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif suffix == '.json':
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")
        
        self.config = ExperimentConfig(**config_dict)
    
    def save_config(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Path where to save configuration
            format: Output format ('yaml' or 'json')
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        if format.lower() == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def get_config(self) -> ExperimentConfig:
        """Get current configuration."""
        if self.config is None:
            raise ValueError("No configuration loaded")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        if self.config is None:
            return False
        
        # Check required paths exist
        required_paths = ['data_path', 'model_path']
        for path_attr in required_paths:
            path = getattr(self.config, path_attr)
            if not Path(path).exists():
                return False
        
        # Check parameter ranges
        if self.config.confidence_threshold < 0 or self.config.confidence_threshold > 1:
            return False
        
        if self.config.ensemble_size < 1:
            return False
        
        if self.config.num_folds < 1:
            return False
        
        return True


def create_default_config(dataset_name: str, output_path: Union[str, Path]) -> None:
    """
    Create a default configuration file for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        output_path: Path where to save configuration
    """
    default_config = ExperimentConfig(
        dataset_name=dataset_name,
        data_path=f"data/{dataset_name}",
        model_path=f"data/models/{dataset_name}_model.pkl",
        output_dir=f"results/{dataset_name}"
    )
    
    config_manager = ConfigManager()
    config_manager.config = default_config
    
    output_path = Path(output_path)
    if output_path.suffix.lower() in ['.yaml', '.yml']:
        config_manager.save_config(output_path, 'yaml')
    elif output_path.suffix.lower() == '.json':
        config_manager.save_config(output_path, 'json')
    else:
        # Default to YAML
        output_path = output_path.with_suffix('.yaml')
        config_manager.save_config(output_path, 'yaml')


def load_dataset_configs(data_dir: Union[str, Path]) -> Dict[str, ExperimentConfig]:
    """
    Load configurations for all datasets in a directory.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary mapping dataset names to configurations
    """
    data_dir = Path(data_dir)
    configs = {}
    
    # Look for configuration files in dataset subdirectories
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            config_files = list(dataset_dir.glob('*.yaml')) + list(dataset_dir.glob('*.yml')) + list(dataset_dir.glob('*.json'))
            
            if config_files:
                # Use the first configuration file found
                config_path = config_files[0]
                try:
                    config_manager = ConfigManager(config_path)
                    if config_manager.validate_config():
                        configs[dataset_dir.name] = config_manager.get_config()
                except Exception as e:
                    print(f"Warning: Failed to load config for {dataset_dir.name}: {e}")
    
    return configs 