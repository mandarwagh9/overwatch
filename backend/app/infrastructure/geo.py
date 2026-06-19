"""Geographic helpers for fusing mobile GPS into the local world frame."""
from __future__ import annotations

import math
from typing import Tuple

EARTH_RADIUS_M = 6_371_000.0


def gps_to_local(
    lat: float, lng: float, ref_lat: float, ref_lng: float
) -> Tuple[float, float]:
    """Equirectangular projection of ``(lat, lng)`` to local ``(x_east, y_north)``
    metres relative to a reference point. Accurate for the small areas a backpack
    multi-camera rig covers; the reference is the local-frame origin.
    """
    d_lat = math.radians(lat - ref_lat)
    d_lng = math.radians(lng - ref_lng)
    x_east = d_lng * math.cos(math.radians(ref_lat)) * EARTH_RADIUS_M
    y_north = d_lat * EARTH_RADIUS_M
    return (x_east, y_north)
