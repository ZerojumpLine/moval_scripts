"""
Data loading utilities for MOVAL experiments.

This module provides common data loading functions used across different
estimation and evaluation modules.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, Dict, Any, Optional


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model from a pickle file.
    
    Args:
        model_path: Path to the model file (.pkl)
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is not a valid pickle file
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.suffix == '.pkl':
        raise ValueError(f"Model file must have .pkl extension: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")


def load_csv_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV data file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def load_numpy_data(npy_path: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array data.
    
    Args:
        npy_path: Path to the .npy file
        
    Returns:
        Loaded numpy array
        
    Raises:
        FileNotFoundError: If .npy file doesn't exist
    """
    npy_path = Path(npy_path)
    
    if not npy_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {npy_path}")
    
    return np.load(npy_path)


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save results to a text file.
    
    Args:
        results: Dictionary containing results to save
        output_path: Path where to save the results
        
    Raises:
        ValueError: If output_path is invalid
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for key, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, np.ndarray):
                f.write(f"{key}: {value.tolist()}\n")
            else:
                f.write(f"{key}: {str(value)}\n")


def get_dataset_paths(data_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Get paths to different dataset components.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary mapping dataset names to their paths
    """
    data_dir = Path(data_dir)
    
    paths = {
        'models': data_dir / 'models',
        'csv_files': data_dir.glob('*.csv'),
        'numpy_files': data_dir.glob('*.npy')
    }
    
    return paths


def validate_data_structure(data_dir: Union[str, Path]) -> bool:
    """
    Validate that the data directory has the expected structure.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    data_dir = Path(data_dir)
    
    required_dirs = ['models']
    required_files = ['*.csv', '*.npy']
    
    # Check required directories
    for dir_name in required_dirs:
        if not (data_dir / dir_name).exists():
            return False
    
    # Check for at least some data files
    has_csv = any(data_dir.glob('*.csv'))
    has_npy = any(data_dir.glob('*.npy'))
    
    return has_csv or has_npy 