"""
NOAA Solar Position Algorithm.

Calculates the sun's altitude (elevation) and azimuth given an observer's
latitude, longitude, and UTC datetime. Based on the NOAA Solar Calculator
spreadsheet formulas.

Accurate for years 1800-2100.

Usage:
    from training.sunlight.sun_position import get_sun_position

    pos = get_sun_position(lat=40.7128, lon=-74.0060, dt=datetime.utcnow())
    print(f"Altitude: {pos['altitude']:.1f}°, Azimuth: {pos['azimuth']:.1f}°")
"""

import math
from datetime import datetime, timezone


def _julian_day(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Day number."""
    y = dt.year
    m = dt.month
    d = dt.day + dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

    if m <= 2:
        y -= 1
        m += 12

    a = int(y / 100)
    b = 2 - a + int(a / 4)

    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _julian_century(jd: float) -> float:
    """Convert Julian Day to Julian Century (from J2000.0 epoch)."""
    return (jd - 2451545.0) / 36525.0


def get_sun_position(lat: float, lon: float, dt: datetime) -> dict:
    """
    Calculate the sun's position for a given observer location and time.

    Args:
        lat: Observer latitude in degrees (positive north)
        lon: Observer longitude in degrees (positive east)
        dt: UTC datetime

    Returns:
        Dictionary with:
            altitude: Sun elevation above horizon in degrees (-90 to 90)
            azimuth: Sun compass bearing in degrees clockwise from north (0-360)
            is_daytime: Whether the sun is above the horizon
    """
    if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    jd = _julian_day(dt)
    jc = _julian_century(jd)

    # Geometric mean longitude of sun (degrees)
    geom_mean_lon = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360

    # Geometric mean anomaly of sun (degrees)
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    geom_mean_anom_rad = math.radians(geom_mean_anom)

    # Eccentricity of earth's orbit
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    # Sun equation of center
    sun_eq_ctr = (
        math.sin(geom_mean_anom_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * geom_mean_anom_rad) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * geom_mean_anom_rad) * 0.000289
    )

    # Sun true longitude and anomaly
    sun_true_lon = geom_mean_lon + sun_eq_ctr

    # Sun apparent longitude
    omega = 125.04 - 1934.136 * jc
    sun_app_lon = sun_true_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    sun_app_lon_rad = math.radians(sun_app_lon)

    # Mean obliquity of the ecliptic
    mean_obliq = (
        23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    )

    # Corrected obliquity
    obliq_corr = mean_obliq + 0.00256 * math.cos(math.radians(omega))
    obliq_corr_rad = math.radians(obliq_corr)

    # Sun declination
    sin_dec = math.sin(obliq_corr_rad) * math.sin(sun_app_lon_rad)
    declination = math.degrees(math.asin(sin_dec))
    dec_rad = math.radians(declination)

    # Equation of time (minutes)
    var_y = math.tan(obliq_corr_rad / 2) ** 2
    geom_mean_lon_rad = math.radians(geom_mean_lon)
    eq_of_time = 4 * math.degrees(
        var_y * math.sin(2 * geom_mean_lon_rad)
        - 2 * ecc * math.sin(geom_mean_anom_rad)
        + 4 * ecc * var_y * math.sin(geom_mean_anom_rad) * math.cos(2 * geom_mean_lon_rad)
        - 0.5 * var_y * var_y * math.sin(4 * geom_mean_lon_rad)
        - 1.25 * ecc * ecc * math.sin(2 * geom_mean_anom_rad)
    )

    # True solar time (minutes)
    time_offset = eq_of_time + 4 * lon
    hours_from_midnight = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    true_solar_time = (hours_from_midnight * 60 + time_offset) % 1440

    # Hour angle (degrees)
    if true_solar_time / 4 < 0:
        hour_angle = true_solar_time / 4 + 180
    else:
        hour_angle = true_solar_time / 4 - 180
    ha_rad = math.radians(hour_angle)

    # Solar zenith and altitude
    lat_rad = math.radians(lat)
    cos_zenith = (
        math.sin(lat_rad) * math.sin(dec_rad)
        + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
    )
    cos_zenith = max(-1, min(1, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    altitude = 90.0 - zenith

    # Solar azimuth
    zenith_rad = math.radians(zenith)
    if zenith_rad != 0:
        cos_azimuth = (
            (math.sin(dec_rad) - math.sin(lat_rad) * math.cos(zenith_rad))
            / (math.cos(lat_rad) * math.sin(zenith_rad))
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))

        if hour_angle > 0:
            azimuth = (360 - azimuth) % 360
    else:
        azimuth = 0.0 if lat >= 0 else 180.0

    return {
        "altitude": round(altitude, 2),
        "azimuth": round(azimuth, 2),
        "is_daytime": altitude > 0,
    }


def get_compass_direction(azimuth: float) -> str:
    """Convert azimuth degrees to a compass direction label."""
    directions = [
        (22.5, "N"), (67.5, "NE"), (112.5, "E"), (157.5, "SE"),
        (202.5, "S"), (247.5, "SW"), (292.5, "W"), (337.5, "NW"),
        (360.1, "N"),
    ]
    for threshold, label in directions:
        if azimuth < threshold:
            return label
    return "N"


if __name__ == "__main__":
    # Example: New York City, right now
    now = datetime.now(timezone.utc)
    pos = get_sun_position(lat=40.7128, lon=-74.0060, dt=now)
    print(f"Time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sun altitude: {pos['altitude']:.1f}°")
    print(f"Sun azimuth:  {pos['azimuth']:.1f}°")
    print(f"Daytime:      {pos['is_daytime']}")
    print(f"Direction:    {get_compass_direction(pos['azimuth'])}")

    # Example: Mumbai, India at noon
    from datetime import datetime
    noon_mumbai = datetime(2026, 6, 15, 6, 30, 0)  # 12:00 IST = 06:30 UTC
    pos = get_sun_position(lat=19.0760, lon=72.8777, dt=noon_mumbai)
    print(f"\nMumbai (Jun 15, noon IST):")
    print(f"Sun altitude: {pos['altitude']:.1f}°")
    print(f"Sun azimuth:  {pos['azimuth']:.1f}°")
    print(f"Direction:    {get_compass_direction(pos['azimuth'])}")
