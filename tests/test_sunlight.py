"""Tests for the sunlight estimation module (sun position + light classifier)."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.sunlight.sun_position import get_sun_position, get_compass_direction
from training.sunlight.light_classifier import (
    classify_light,
    estimate_lux_from_ev,
    compute_ev_from_exif,
    LightAssessment,
)


class TestSunPosition:
    """Tests for the NOAA sun position algorithm."""

    def test_sun_above_horizon_at_noon_nyc_summer(self):
        dt = datetime(2026, 6, 21, 17, 0, 0)  # 12 PM EST = 17:00 UTC
        pos = get_sun_position(lat=40.7128, lon=-74.0060, dt=dt)
        assert pos["altitude"] > 50, f"Expected altitude > 50°, got {pos['altitude']}"
        assert pos["is_daytime"] is True

    def test_sun_below_horizon_at_midnight_nyc(self):
        dt = datetime(2026, 6, 21, 5, 0, 0)  # midnight EST = 05:00 UTC
        pos = get_sun_position(lat=40.7128, lon=-74.0060, dt=dt)
        assert pos["altitude"] < 0, f"Expected altitude < 0°, got {pos['altitude']}"
        assert pos["is_daytime"] is False

    def test_equatorial_noon_high_altitude(self):
        dt = datetime(2026, 3, 20, 12, 0, 0)  # equinox noon UTC
        pos = get_sun_position(lat=0.0, lon=0.0, dt=dt)
        assert pos["altitude"] > 70, f"Expected altitude > 70°, got {pos['altitude']}"

    def test_azimuth_range(self):
        dt = datetime(2026, 6, 21, 17, 0, 0)
        pos = get_sun_position(lat=40.7128, lon=-74.0060, dt=dt)
        assert 0 <= pos["azimuth"] <= 360

    def test_mumbai_noon_summer(self):
        dt = datetime(2026, 6, 15, 6, 30, 0)  # noon IST = 06:30 UTC
        pos = get_sun_position(lat=19.0760, lon=72.8777, dt=dt)
        assert pos["altitude"] > 60, f"Expected altitude > 60°, got {pos['altitude']}"
        assert pos["is_daytime"] is True


class TestCompassDirection:
    """Tests for compass direction conversion."""

    def test_north(self):
        assert get_compass_direction(0) == "N"
        assert get_compass_direction(350) == "N"
        assert get_compass_direction(10) == "N"

    def test_south(self):
        assert get_compass_direction(180) == "S"

    def test_east(self):
        assert get_compass_direction(90) == "E"

    def test_west(self):
        assert get_compass_direction(270) == "W"

    def test_northeast(self):
        assert get_compass_direction(45) == "NE"

    def test_all_eight_directions(self):
        expected = {0: "N", 45: "NE", 90: "E", 135: "SE",
                    180: "S", 225: "SW", 270: "W", 315: "NW"}
        for azimuth, direction in expected.items():
            assert get_compass_direction(azimuth) == direction


class TestLuxEstimation:
    """Tests for camera-based lux estimation."""

    def test_bright_sunlight_ev(self):
        lux = estimate_lux_from_ev(15.0)
        assert lux > 10_000, f"EV 15 should be bright sun, got {lux} lux"

    def test_indoor_lighting_ev(self):
        lux = estimate_lux_from_ev(8.0)
        assert 200 < lux < 2000, f"EV 8 should be indoor level, got {lux} lux"

    def test_dim_room_ev(self):
        lux = estimate_lux_from_ev(5.0)
        assert lux < 200, f"EV 5 should be dim, got {lux} lux"

    def test_ev_from_exif(self):
        ev = compute_ev_from_exif(f_number=2.4, exposure_time=1/1000, iso=100)
        assert 10 < ev < 15, f"Outdoor EXIF should give EV 10-15, got {ev}"


class TestLightClassifier:
    """Tests for the fusion-based light classifier."""

    def test_direct_sunlight(self):
        result = classify_light(
            sun_altitude=45.0, sun_azimuth=180.0,
            camera_heading=170.0, estimated_lux=15000.0, latitude=40.7
        )
        assert result.category == "Direct Sunlight"
        assert result.facing_sun is True

    def test_bright_indirect(self):
        result = classify_light(
            sun_altitude=45.0, sun_azimuth=180.0,
            camera_heading=90.0, estimated_lux=5000.0, latitude=40.7
        )
        assert result.category == "Bright Indirect"

    def test_medium_light(self):
        result = classify_light(
            sun_altitude=30.0, sun_azimuth=180.0,
            camera_heading=0.0, estimated_lux=1000.0, latitude=40.7
        )
        assert result.category == "Medium Light"

    def test_low_light(self):
        result = classify_light(
            sun_altitude=20.0, sun_azimuth=180.0,
            camera_heading=0.0, estimated_lux=200.0, latitude=40.7
        )
        assert result.category == "Low Light / Shade"

    def test_nighttime(self):
        result = classify_light(
            sun_altitude=-10.0, sun_azimuth=0.0,
            camera_heading=180.0, estimated_lux=50.0, latitude=40.7
        )
        assert "Nighttime" in result.category

    def test_suitable_plants_returned(self):
        result = classify_light(
            sun_altitude=45.0, sun_azimuth=180.0,
            camera_heading=170.0, estimated_lux=15000.0, latitude=40.7
        )
        assert len(result.suitable_plants) > 0

    def test_to_dict(self):
        result = classify_light(
            sun_altitude=45.0, sun_azimuth=180.0,
            camera_heading=170.0, estimated_lux=15000.0, latitude=40.7
        )
        d = result.to_dict()
        assert "light_category" in d
        assert "compass_direction" in d
        assert "suitable_plants" in d

    def test_southern_hemisphere(self):
        result = classify_light(
            sun_altitude=45.0, sun_azimuth=0.0,
            camera_heading=5.0, estimated_lux=15000.0, latitude=-33.8
        )
        assert result.category == "Direct Sunlight"
        assert result.compass_direction == "N"
