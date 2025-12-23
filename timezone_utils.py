# timezone_utils.py
"""
Timezone utilities for the trading bot.
Ensures all datetime operations use the configured timezone.
"""
import os
import pytz
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Get timezone from environment or default to UTC
TIMEZONE_STR = os.getenv("TIMEZONE", "UTC")

try:
    TIMEZONE = pytz.timezone(TIMEZONE_STR)
    logger.info(f"✅ Timezone set to: {TIMEZONE_STR}")
except pytz.exceptions.UnknownTimeZoneError:
    logger.warning(f"⚠️ Unknown timezone '{TIMEZONE_STR}', falling back to UTC")
    TIMEZONE = pytz.UTC
    TIMEZONE_STR = "UTC"


def now() -> datetime:
    """
    Get current time in the configured timezone.
    
    Returns:
        datetime: Current time with timezone info
    """
    return datetime.now(TIMEZONE)


def localize(dt: datetime) -> datetime:
    """
    Add timezone info to a naive datetime object.
    
    Args:
        dt: Naive datetime object
    
    Returns:
        datetime: Timezone-aware datetime
    """
    if dt.tzinfo is None:
        return TIMEZONE.localize(dt)
    return dt


def to_timezone(dt: datetime, tz: Optional[pytz.timezone] = None) -> datetime:
    """
    Convert datetime to specified timezone.
    
    Args:
        dt: Datetime to convert
        tz: Target timezone (defaults to configured TIMEZONE)
    
    Returns:
        datetime: Converted datetime
    """
    if tz is None:
        tz = TIMEZONE
    
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    
    return dt.astimezone(tz)


def parse_time_12h(hour: int, minute: int, am_pm: str) -> datetime:
    """
    Parse 12-hour time format to datetime in configured timezone.
    
    Args:
        hour: Hour (1-12)
        minute: Minute (0-59)
        am_pm: "AM" or "PM"
    
    Returns:
        datetime: Parsed time in configured timezone
    """
    # Convert to 24-hour format
    if am_pm.upper() == "PM" and hour != 12:
        hour += 12
    elif am_pm.upper() == "AM" and hour == 12:
        hour = 0
    
    # Create datetime for today
    current = now()
    entry_time = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # If time is in the past...
    if entry_time < current:
        delta = (current - entry_time).total_seconds()
        # If signal is less than 30 mins late, assume it's for TODAY (just delayed)
        if delta < 1800: 
            pass 
        else:
            # Otherwise assume it's scheduled for tomorrow
            from datetime import timedelta
            entry_time += timedelta(days=1)
    
    return entry_time


def format_time(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Format datetime with timezone info.
    
    Args:
        dt: Datetime to format
        format_str: Format string
    
    Returns:
        str: Formatted datetime string
    """
    if dt.tzinfo is None:
        dt = localize(dt)
    
    return dt.strftime(format_str)


def get_timezone_name() -> str:
    """Get the configured timezone name."""
    return TIMEZONE_STR


def get_timezone() -> pytz.timezone:
    """Get the configured timezone object."""
    return TIMEZONE
