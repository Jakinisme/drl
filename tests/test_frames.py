"""Round-trip and sign-convention tests for common/frames.py."""

import numpy as np
import pytest

from common.frames import (
    EARTH_RADIUS_M,
    body_to_enu,
    enu_to_body,
    enu_to_latlon,
    goal_in_body,
    heading_deg_to_psi,
    heading_error_deg,
    heading_error_rad,
    latlon_to_enu,
    psi_to_heading_deg,
    wrap_deg,
    wrap_pi,
)

LAT0, LON0 = -6.200000, 106.816666  # Jakarta-ish, arbitrary


def test_heading_deg_rad_roundtrip():
    hdg = np.array([0.0, 45.0, 155.0, 180.0, 270.0, 359.9])
    np.testing.assert_allclose(psi_to_heading_deg(heading_deg_to_psi(hdg)), hdg, atol=1e-9)


def test_psi_zero_bow_north():
    psi = 0.0
    east, north = body_to_enu(1.0, 0.0, psi)
    assert abs(east) < 1e-12
    np.testing.assert_allclose(north, 1.0)


def test_psi_90_bow_east():
    east, north = body_to_enu(1.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(east, 1.0, atol=1e-12)
    np.testing.assert_allclose(north, 0.0, atol=1e-12)


def test_starboard_points_east_when_bow_north():
    east, north = body_to_enu(0.0, 1.0, 0.0)
    np.testing.assert_allclose(east, 1.0, atol=1e-12)
    np.testing.assert_allclose(north, 0.0, atol=1e-12)


@pytest.mark.parametrize("psi_deg", [0.0, 37.0, 90.0, 180.0, 251.3, 359.0])
def test_body_enu_roundtrip(psi_deg):
    psi = np.deg2rad(psi_deg)
    u, v = 0.8, -0.3
    east, north = body_to_enu(u, v, psi)
    u2, v2 = enu_to_body(east, north, psi)
    np.testing.assert_allclose([u2, v2], [u, v], atol=1e-12)
    assert abs(np.hypot(east, north) - np.hypot(u, v)) < 1e-12


def test_latlon_enu_roundtrip():
    lat = np.array([LAT0, LAT0 + 0.001, LAT0 - 0.002])
    lon = np.array([LON0, LON0 - 0.0015, LON0 + 0.0007])
    east, north = latlon_to_enu(lat, lon, LAT0, LON0)
    np.testing.assert_allclose(east[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(north[0], 0.0, atol=1e-9)
    lat2, lon2 = enu_to_latlon(east, north, LAT0, LON0)
    np.testing.assert_allclose(lat2, lat, atol=1e-12)
    np.testing.assert_allclose(lon2, lon, atol=1e-12)


def test_latlon_matches_geo_py_flat_earth():
    """Same numbers as pi-control/src/utils/geo.py lat_lon_offset()."""
    north_m, east_m = 100.0, -50.0
    lat = LAT0 + (north_m / EARTH_RADIUS_M) * (180 / np.pi)
    lon = LON0 + (east_m / (EARTH_RADIUS_M * np.cos(np.deg2rad(LAT0)))) * (180 / np.pi)
    east, north = latlon_to_enu(lat, lon, LAT0, LON0)
    np.testing.assert_allclose([east, north], [east_m, north_m], rtol=1e-9, atol=1e-7)


def test_goal_in_body_ahead():
    dx, dy = goal_in_body(10.0, 5.0, 10.0, 0.0, 0.0)  # goal 5 m north, bow north
    np.testing.assert_allclose([dx, dy], [5.0, 0.0], atol=1e-12)


def test_goal_in_body_to_starboard():
    dx, dy = goal_in_body(5.0, 0.0, 0.0, 0.0, 0.0)  # goal 5 m east, bow north
    np.testing.assert_allclose([dx, dy], [0.0, 5.0], atol=1e-12)


def test_heading_error_sign():
    assert heading_error_deg(350.0, 10.0) == pytest.approx(20.0)
    assert heading_error_deg(10.0, 350.0) == pytest.approx(-20.0)
    assert abs(heading_error_deg(0.0, 180.0)) == pytest.approx(180.0)  # 180 deg is ambiguous, wraps to -180
    np.testing.assert_allclose(heading_error_rad(np.deg2rad(350.0), np.deg2rad(10.0)),
                               np.deg2rad(20.0), atol=1e-12)


def test_wrap():
    np.testing.assert_allclose(wrap_deg([-10.0, 370.0, 360.0]), [350.0, 10.0, 0.0])
    np.testing.assert_allclose(wrap_pi([3 * np.pi, -np.pi, 0.0]), [-np.pi, -np.pi, 0.0], atol=1e-12)
    assert -np.pi <= wrap_pi(np.pi) < np.pi


def test_broadcast_shapes():
    psi = np.zeros((4, 3))
    u, v = enu_to_body(np.ones((4, 3)), np.zeros((4, 3)), psi)
    assert u.shape == (4, 3) and v.shape == (4, 3)
    np.testing.assert_allclose(u, 0.0)
