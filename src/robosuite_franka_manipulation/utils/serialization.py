"""
Serialization utilities for ZMQ services and data exchange.

This module provides helper functions for converting between Python/NumPy
types and JSON-serializable formats.
"""

import numpy as np
from typing import Dict, List, Any, Union


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to JSON-serializable format.
    
    Handles:
    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - dicts -> recursively converted dicts
    - lists -> recursively converted lists
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    else:
        return obj


def from_json_to_numpy(obj: Any, convert_lists: bool = False) -> Any:
    """
    Convert JSON data back to numpy arrays where appropriate.
    
    Args:
        obj: JSON-loaded object
        convert_lists: If True, convert all lists to numpy arrays
        
    Returns:
        Object with lists converted to numpy arrays where appropriate
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Known keys that should be numpy arrays
            if key in ('pos', 'position', 'quat', 'quaternion', 'pcd', 'pointcloud'):
                if isinstance(value, list):
                    result[key] = np.array(value)
                else:
                    result[key] = value
            else:
                result[key] = from_json_to_numpy(value, convert_lists)
        return result
    elif isinstance(obj, list):
        if convert_lists:
            return np.array(obj)
        return [from_json_to_numpy(item, convert_lists) for item in obj]
    else:
        return obj


def convert_grasps_to_json(grasps: List[Dict]) -> List[Dict]:
    """
    Convert a list of grasp dictionaries to JSON-serializable format.
    
    Args:
        grasps: List of grasp dictionaries with numpy arrays
        
    Returns:
        List of grasp dictionaries with JSON-serializable values
    """
    return [to_json_serializable(grasp) for grasp in grasps]


def convert_detections_to_json(detections: List[Dict]) -> List[Dict]:
    """
    Convert a list of detection dictionaries to JSON-serializable format.
    
    Args:
        detections: List of detection dictionaries with numpy arrays
        
    Returns:
        List of detection dictionaries with JSON-serializable values
    """
    return [to_json_serializable(detection) for detection in detections]


def convert_pose_to_json(pose: Dict) -> Dict:
    """
    Convert a pose dictionary to JSON-serializable format.
    
    Args:
        pose: Pose dictionary with 'pos' and 'quat' keys
        
    Returns:
        JSON-serializable pose dictionary
    """
    return to_json_serializable(pose)


def numpy_to_list(arr: Union[np.ndarray, List]) -> List:
    """
    Convert numpy array to list, handling nested arrays.
    
    Args:
        arr: Numpy array or list
        
    Returns:
        List representation
    """
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def ensure_numpy(arr: Union[np.ndarray, List], dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Ensure input is a numpy array with specified dtype.
    
    Args:
        arr: Input array or list
        dtype: Desired numpy dtype
        
    Returns:
        Numpy array
    """
    if isinstance(arr, np.ndarray):
        return arr.astype(dtype)
    return np.array(arr, dtype=dtype)
