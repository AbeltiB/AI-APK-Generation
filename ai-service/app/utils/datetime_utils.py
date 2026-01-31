"""
ai-service/app/utils/datetime_utils.py
Production-ready timezone-aware datetime utilities.
"""
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


def to_iso_string(dt: Optional[datetime] = None) -> str:
    """
    Convert datetime to ISO 8601 string with Z suffix.
    
    Args:
        dt: Datetime object (defaults to now)
        
    Returns:
        ISO 8601 formatted string ending with 'Z'
        
    Example:
        >>> to_iso_string()
        '2025-01-15T10:30:45.123Z'
    """
    if dt is None:
        dt = utc_now()
    
    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Format with milliseconds and replace +00:00 with Z
    iso_str = dt.isoformat(timespec='milliseconds')
    if iso_str.endswith('+00:00'):
        return iso_str[:-6] + 'Z'
    return iso_str


def from_iso_string(iso_string: str) -> datetime:
    """
    Parse ISO 8601 string to timezone-aware datetime.
    
    Args:
        iso_string: ISO 8601 formatted string
        
    Returns:
        Timezone-aware datetime object
    """
    # Handle Z suffix
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1] + '+00:00'
    
    dt = datetime.fromisoformat(iso_string)
    
    # Ensure UTC if no timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(utc_now().timestamp() * 1000)


def age_seconds(dt: datetime) -> float:
    """
    Calculate age of datetime in seconds from now.
    
    Args:
        dt: Past datetime
        
    Returns:
        Age in seconds
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return (utc_now() - dt).total_seconds()


# Backward compatibility
def utcnow() -> datetime:
    """
    Deprecated: Use utc_now() instead.
    Kept for backward compatibility.
    """
    return utc_now()