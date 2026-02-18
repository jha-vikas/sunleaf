"""
Light condition classifier using sensor fusion.

Combines:
  1. Sun position (from NOAA algorithm via GPS + time)
  2. Camera-derived brightness (from EXIF exposure metadata)
  3. Compass heading (which direction the camera is pointing)

To classify indoor light conditions for plant placement:
  - Direct Sunlight (>10,800 lux)
  - Bright Indirect (2,700 - 10,800 lux)
  - Medium Light (500 - 2,700 lux)
  - Low Light / Shade (<500 lux)

Usage:
    from training.sunlight.light_classifier import classify_light, estimate_lux_from_ev

    lux = estimate_lux_from_ev(ev=12.0)
    result = classify_light(
        sun_altitude=45.0, sun_azimuth=180.0,
        camera_heading=170.0, estimated_lux=lux,
        latitude=40.7
    )
"""

from dataclasses import dataclass

from .sun_position import get_compass_direction


# Lux thresholds for plant light categories
# Based on horticultural standards (foot-candles converted to lux)
LUX_DIRECT_SUN = 10_800       # >1000 fc: direct sunbeams on leaves
LUX_BRIGHT_INDIRECT = 2_700   # 250-1000 fc: bright but no direct rays
LUX_MEDIUM = 500              # 50-250 fc: readable but not bright
# Below 500 lux = low light


@dataclass
class LightAssessment:
    category: str
    estimated_lux: float
    compass_direction: str
    facing_sun: bool
    sun_altitude: float
    sun_azimuth: float
    recommendation: str
    daily_light_estimate: str
    suitable_plants: list[str]

    def to_dict(self) -> dict:
        return {
            "light_category": self.category,
            "estimated_lux": self.estimated_lux,
            "compass_direction": self.compass_direction,
            "facing_sun": self.facing_sun,
            "sun_altitude": self.sun_altitude,
            "sun_azimuth": self.sun_azimuth,
            "recommendation": self.recommendation,
            "daily_light_estimate": self.daily_light_estimate,
            "suitable_plants": self.suitable_plants,
        }


def estimate_lux_from_ev(ev: float) -> float:
    """
    Estimate illuminance (lux) from camera Exposure Value (EV).

    EV can be computed from camera EXIF data:
        EV = log2(f_number^2 / exposure_time) - log2(ISO / 100)

    The relationship between EV (at ISO 100) and lux is approximately:
        Lux ≈ 2.5 * 2^EV
    """
    return 2.5 * (2.0 ** ev)


def compute_ev_from_exif(f_number: float, exposure_time: float, iso: int) -> float:
    """
    Compute Exposure Value from EXIF camera parameters.

    Args:
        f_number: Aperture f-number (e.g., 1.8, 2.4)
        exposure_time: Shutter speed in seconds (e.g., 1/60 = 0.0167)
        iso: ISO sensitivity (e.g., 100, 400)
    """
    import math
    ev_at_iso = math.log2(f_number ** 2 / exposure_time)
    ev_100 = ev_at_iso + math.log2(iso / 100)
    return ev_100


def _angle_difference(a: float, b: float) -> float:
    """Compute the smallest angle between two compass bearings (0-180)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _get_window_direction_info(compass_direction: str, latitude: float) -> tuple[str, str]:
    """
    Estimate daily light based on window direction and hemisphere.

    Returns:
        (daily_light_estimate, recommendation)
    """
    is_northern_hemisphere = latitude >= 0

    direction_map_north = {
        "S":  ("6-8 hours of direct + indirect light",
               "Excellent for most plants. Best spot for sun-loving species."),
        "SE": ("5-7 hours, strong morning to midday light",
               "Great for most tropical and flowering plants."),
        "SW": ("5-7 hours, midday to afternoon light",
               "Good for sun-lovers; may get hot afternoon sun in summer."),
        "E":  ("3-5 hours of gentle morning light",
               "Ideal for plants that like bright indirect light. Morning sun is gentle."),
        "W":  ("3-5 hours of afternoon light",
               "Good light but hot afternoon sun may scorch sensitive leaves."),
        "NE": ("2-4 hours of soft morning light",
               "Suitable for medium-light plants. Gentle, indirect conditions."),
        "NW": ("2-4 hours of soft afternoon light",
               "Moderate light. Good for plants tolerant of medium conditions."),
        "N":  ("1-2 hours of indirect light only",
               "Low light. Best for shade-tolerant plants like pothos and ZZ plants."),
    }

    # In Southern Hemisphere, swap N<->S logic
    direction_map_south = {
        "N":  direction_map_north["S"],
        "NE": direction_map_north["SE"],
        "NW": direction_map_north["SW"],
        "E":  direction_map_north["E"],
        "W":  direction_map_north["W"],
        "SE": direction_map_north["NE"],
        "SW": direction_map_north["NW"],
        "S":  direction_map_north["N"],
    }

    lookup = direction_map_north if is_northern_hemisphere else direction_map_south
    return lookup.get(compass_direction, ("Unknown", "Point camera toward a window for best results."))


PLANTS_BY_LIGHT = {
    "Direct Sunlight": [
        "Aloe Vera", "Jade Plant", "Bird of Paradise",
        "Sago Palm", "Yucca", "Poinsettia",
    ],
    "Bright Indirect": [
        "Monstera Deliciosa", "Pothos", "Rubber Plant",
        "Boston Fern", "Calathea", "Peace Lily",
    ],
    "Medium Light": [
        "Snake Plant", "ZZ Plant", "Dracaena",
        "Chinese Evergreen", "Parlor Palm", "Cast Iron Plant",
    ],
    "Low Light / Shade": [
        "ZZ Plant", "Snake Plant", "Pothos",
        "Cast Iron Plant", "Chinese Evergreen",
    ],
}


def classify_light(
    sun_altitude: float,
    sun_azimuth: float,
    camera_heading: float,
    estimated_lux: float,
    latitude: float = 0.0,
) -> LightAssessment:
    """
    Classify the light condition at the camera's location.

    Args:
        sun_altitude: Sun elevation in degrees (from NOAA algorithm)
        sun_azimuth: Sun compass bearing in degrees
        camera_heading: Camera/phone compass heading in degrees
        estimated_lux: Estimated illuminance from camera EXIF
        latitude: Observer latitude (for hemisphere-aware recommendations)
    """
    angle_to_sun = _angle_difference(camera_heading, sun_azimuth)
    facing_sun = angle_to_sun < 45

    compass_direction = get_compass_direction(camera_heading)

    if sun_altitude <= 0:
        category = "No Natural Light (Nighttime)"
        daily_est, rec = _get_window_direction_info(compass_direction, latitude)
        rec = f"Nighttime reading. During the day: {rec}"
    elif facing_sun and estimated_lux > LUX_DIRECT_SUN:
        category = "Direct Sunlight"
        daily_est, rec = _get_window_direction_info(compass_direction, latitude)
    elif estimated_lux > LUX_BRIGHT_INDIRECT:
        category = "Bright Indirect"
        daily_est, rec = _get_window_direction_info(compass_direction, latitude)
    elif estimated_lux > LUX_MEDIUM:
        category = "Medium Light"
        daily_est, rec = _get_window_direction_info(compass_direction, latitude)
    else:
        category = "Low Light / Shade"
        daily_est, rec = _get_window_direction_info(compass_direction, latitude)

    plants = PLANTS_BY_LIGHT.get(category, PLANTS_BY_LIGHT["Low Light / Shade"])

    return LightAssessment(
        category=category,
        estimated_lux=round(estimated_lux, 0),
        compass_direction=compass_direction,
        facing_sun=facing_sun,
        sun_altitude=sun_altitude,
        sun_azimuth=sun_azimuth,
        recommendation=rec,
        daily_light_estimate=daily_est,
        suitable_plants=plants,
    )


if __name__ == "__main__":
    import json

    # Simulate: south-facing window, bright afternoon in Northern Hemisphere
    result = classify_light(
        sun_altitude=45.0,
        sun_azimuth=180.0,
        camera_heading=170.0,
        estimated_lux=15000.0,
        latitude=40.7,
    )
    print("Scenario: South-facing window, bright afternoon")
    print(json.dumps(result.to_dict(), indent=2))

    # Simulate: north-facing window, same time
    result2 = classify_light(
        sun_altitude=45.0,
        sun_azimuth=180.0,
        camera_heading=5.0,
        estimated_lux=300.0,
        latitude=40.7,
    )
    print("\nScenario: North-facing window")
    print(json.dumps(result2.to_dict(), indent=2))

    # Simulate: EV-based lux estimation from camera EXIF
    ev = compute_ev_from_exif(f_number=2.4, exposure_time=1 / 1000, iso=100)
    lux = estimate_lux_from_ev(ev)
    print(f"\nEXIF: f/2.4, 1/1000s, ISO 100 -> EV={ev:.1f} -> Lux={lux:.0f}")
