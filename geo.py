"""
Geo utilities for the accessibility backend.
Contains helper functions for geographic calculations and data loading.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional
from shapely.geometry import Point, shape


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def load_json(path: str | Path) -> Any:
    """
    Load and parse a JSON file.
    
    Args:
        path: Path to the JSON file
    
    Returns:
        Parsed JSON data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    """
    Save data to a JSON file.
    
    Args:
        path: Path to the JSON file
        data: Data to save
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_grid_for_point(lat: float, lon: float, grid_features: list[dict]) -> Optional[int]:
    """
    Find which grid polygon contains the given point.
    Uses Shapely for point-in-polygon test.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        grid_features: List of GeoJSON features with polygon geometries
    
    Returns:
        grid_id if found, None otherwise
    """
    point = Point(lon, lat)  # Note: GeoJSON uses (lon, lat) order
    
    for feature in grid_features:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return feature['properties']['grid_id']
    
    return None


def find_nearest_grid(lat: float, lon: float, grid_features: list[dict]) -> Optional[int]:
    """
    Find the nearest grid by centroid distance.
    Fallback method if point-in-polygon fails.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        grid_features: List of GeoJSON features with polygon geometries
    
    Returns:
        grid_id of nearest grid
    """
    min_distance = float('inf')
    nearest_grid_id = None
    
    for feature in grid_features:
        polygon = shape(feature['geometry'])
        centroid = polygon.centroid
        distance = haversine_distance(lat, lon, centroid.y, centroid.x)
        
        if distance < min_distance:
            min_distance = distance
            nearest_grid_id = feature['properties']['grid_id']
    
    return nearest_grid_id


def calculate_walking_duration(distance_m: float, speed_ms: float = 1.4) -> float:
    """
    Calculate walking duration in minutes.
    
    Args:
        distance_m: Distance in meters
        speed_ms: Walking speed in m/s (default: 1.4 m/s)
    
    Returns:
        Duration in minutes
    """
    return distance_m / speed_ms / 60

