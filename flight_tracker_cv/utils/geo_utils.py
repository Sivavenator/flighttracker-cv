"""
Geospatial utility functions for FlightTracker CV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LatLon:
    lat: float
    lon: float

    def __str__(self) -> str:
        lat_dir = "N" if self.lat >= 0 else "S"
        lon_dir = "E" if self.lon >= 0 else "W"
        return f"{abs(self.lat):.4f}°{lat_dir} {abs(self.lon):.4f}°{lon_dir}"


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using the Haversine formula.
    Returns distance in kilometres.
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------

def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.
    Returns degrees in [0, 360) where 0 = North, 90 = East.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Altitude estimate
# ---------------------------------------------------------------------------

def estimate_altitude_from_size(
    bbox_height_px: float,
    frame_height_px: float,
    vfov_deg: float = 45.0,
    known_wingspan_m: float = 60.0,
) -> float:
    """
    Estimate aircraft altitude (metres above camera) from apparent bounding-box
    height using a pin-hole camera model.

    Returns altitude in metres (very rough estimate).
    """
    if bbox_height_px <= 0 or frame_height_px <= 0:
        return -1.0
    # Focal length in pixels from VFOV
    fl_px = (frame_height_px / 2) / math.tan(math.radians(vfov_deg / 2))
    # Apparent angular size
    angular_height_rad = math.atan(bbox_height_px / fl_px)
    if angular_height_rad <= 0:
        return -1.0
    # Approximate wingspan visible ~ height; use wingspan as reference
    altitude_m = known_wingspan_m / math.tan(angular_height_rad)
    return round(altitude_m, 0)


# ---------------------------------------------------------------------------
# Distance estimate
# ---------------------------------------------------------------------------

def estimate_distance_from_size(
    bbox_width_px: float,
    frame_width_px: float,
    hfov_deg: float = 60.0,
    known_wingspan_m: float = 60.0,
) -> float:
    """
    Estimate line-of-sight distance to aircraft in metres using apparent wingspan.
    """
    if bbox_width_px <= 0 or frame_width_px <= 0:
        return -1.0
    fl_px = (frame_width_px / 2) / math.tan(math.radians(hfov_deg / 2))
    distance_m = (known_wingspan_m * fl_px) / bbox_width_px
    return round(distance_m, 0)


# ---------------------------------------------------------------------------
# Bearing from image position
# ---------------------------------------------------------------------------

def bearing_from_image_position(
    bbox_cx: float,
    frame_width: int,
    hfov_deg: float = 60.0,
    camera_heading_deg: float = 0.0,
) -> float:
    """
    Convert a bounding-box centre x pixel coordinate to compass bearing.

    bbox_cx: horizontal centre of the bounding box in pixels
    frame_width: total frame width in pixels
    hfov_deg: camera horizontal field of view in degrees
    camera_heading_deg: direction camera is pointed (0 = North)

    Returns bearing in [0, 360).
    """
    offset_from_centre = bbox_cx - (frame_width / 2.0)
    degrees_per_px = hfov_deg / frame_width
    offset_deg = offset_from_centre * degrees_per_px
    return (camera_heading_deg + offset_deg) % 360.0


# ---------------------------------------------------------------------------
# Sun position (simplified)
# ---------------------------------------------------------------------------

def get_sun_position(
    lat: float,
    lon: float,
    dt: datetime,
) -> Tuple[float, float]:
    """
    Compute approximate solar altitude and azimuth for a given location + time.

    Uses NOAA simplified algorithm. Accuracy ~1 degree.
    Returns (altitude_deg, azimuth_deg).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Julian Day Number
    jd = _julian_day(dt)
    n = jd - 2451545.0

    # Solar mean longitude (deg)
    L = (280.460 + 0.9856474 * n) % 360
    # Solar mean anomaly (deg)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    # Ecliptic longitude (deg)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    # Obliquity of the ecliptic
    eps = math.radians(23.439 - 0.0000004 * n)

    # Right ascension, declination
    sin_dec = math.sin(eps) * math.sin(lam)
    dec = math.asin(sin_dec)

    # Hour angle
    ut_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    gmst = (6.697375 + 0.0657098242 * n + ut_hours) % 24  # GMST in hours
    lmst = (gmst + lon / 15.0) % 24
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    ha_rad = math.radians((lmst - math.degrees(ra) / 15.0) * 15.0)

    lat_r = math.radians(lat)
    # Altitude
    sin_alt = (math.sin(lat_r) * math.sin(dec)
               + math.cos(lat_r) * math.cos(dec) * math.cos(ha_rad))
    altitude_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

    # Azimuth
    cos_az = (math.sin(dec) - math.sin(lat_r) * sin_alt) / (
        math.cos(lat_r) * math.cos(math.radians(altitude_deg)) + 1e-9
    )
    azimuth_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(ha_rad) > 0:
        azimuth_deg = 360 - azimuth_deg

    return round(altitude_deg, 2), round(azimuth_deg, 2)


def _julian_day(dt: datetime) -> float:
    """Convert datetime to Julian Day Number."""
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd + h / 24.0


# ---------------------------------------------------------------------------
# Daytime check
# ---------------------------------------------------------------------------

def is_daytime(lat: float, lon: float, dt: Optional[datetime] = None) -> bool:
    """Return True if the sun is above the horizon at the given location and time."""
    from datetime import datetime as _dt
    if dt is None:
        dt = _dt.now(timezone.utc)
    altitude, _ = get_sun_position(lat, lon, dt)
    return altitude > 0.0


# Fix missing import
from typing import Optional  # noqa: E402


# ---------------------------------------------------------------------------
# Cardinal direction
# ---------------------------------------------------------------------------

class CardinalDirection:
    """Convert a bearing in degrees to a 16-point compass direction string."""

    _DIRS = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]

    @classmethod
    def from_degrees(cls, deg: float) -> str:
        deg = deg % 360
        idx = int((deg + 11.25) / 22.5) % 16
        return cls._DIRS[idx]
