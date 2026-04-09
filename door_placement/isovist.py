"""
Isovist (Visibility Polygon) Engine for 2D floor plans.

Computes the set of all points visible from a given observer position,
respecting wall occlusions.  This is the gold-standard technique in
architectural spatial analysis (Space Syntax / Benedikt 1979).

Algorithm — Angular Sweep:
    1. Collect all wall-segment endpoints.
    2. For each endpoint, cast 3 rays (at, ε-left, ε-right).
    3. Sort rays by angle.
    4. For each ray, find the nearest wall intersection.
    5. Connect the nearest-hit points → visibility polygon (isovist).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, LineString, Point


# ── Types ───────────────────────────────────────────────────────────────
Vec2  = Tuple[float, float]
Seg   = Tuple[Vec2, Vec2]          # (p1, p2) wall segment


# ── Public API ──────────────────────────────────────────────────────────

def compute_isovist(
    origin: Vec2,
    wall_segments: List[Seg],
    max_radius: float = 1e6,
) -> Polygon:
    """Compute the full 360° isovist (visibility polygon) from *origin*.

    Parameters
    ----------
    origin : (x, y)
        Observer position.
    wall_segments : list of ((x1,y1),(x2,y2))
        Every wall segment in the floor plan.
    max_radius : float
        Maximum sight distance.  Rays that don't hit any wall stop here.

    Returns
    -------
    Polygon
        The visibility polygon.
    """
    pts = _angular_sweep(origin, wall_segments, max_radius)
    if len(pts) < 3:
        return Point(origin).buffer(1)  # degenerate
    return Polygon(pts)


def compute_directional_isovist(
    origin: Vec2,
    direction: Vec2,
    fov_deg: float,
    wall_segments: List[Seg],
    max_radius: float = 1e6,
) -> Polygon:
    """Compute a directional isovist (limited field-of-view).

    Parameters
    ----------
    origin : (x, y)
    direction : (dx, dy)
        Unit vector of the gaze direction.
    fov_deg : float
        Total field-of-view in degrees (e.g. 120° = ±60° from centre).
    wall_segments : list of segments
    max_radius : float

    Returns
    -------
    Polygon
    """
    dx, dy = direction
    centre_angle = math.atan2(dy, dx)
    half_fov = math.radians(fov_deg / 2)
    lo = centre_angle - half_fov
    hi = centre_angle + half_fov

    pts = _angular_sweep(origin, wall_segments, max_radius,
                         angle_lo=lo, angle_hi=hi)
    if len(pts) < 2:
        return Point(origin).buffer(1)
    # Add origin to close the pie-slice
    pts = [origin] + pts
    if len(pts) < 3:
        return Point(origin).buffer(1)
    return Polygon(pts)


# ── Core engine ─────────────────────────────────────────────────────────

def _angular_sweep(
    origin: Vec2,
    wall_segments: List[Seg],
    max_radius: float,
    angle_lo: Optional[float] = None,
    angle_hi: Optional[float] = None,
) -> List[Vec2]:
    """Angular-sweep raycasting.

    1.  Collect unique angles to every wall endpoint, ±ε.
    2.  Sort angles.
    3.  For each angle, cast a ray and record the nearest hit.
    """
    ox, oy = origin
    epsilon = 1e-5

    # ── 1. Collect all unique target angles ──────────────────────────────
    angles: List[float] = []
    for (ax, ay), (bx, by) in wall_segments:
        for px, py in ((ax, ay), (bx, by)):
            a = math.atan2(py - oy, px - ox)
            angles.extend([a - epsilon, a, a + epsilon])

    if angle_lo is not None and angle_hi is not None:
        # Filter to the requested angular range
        angles = [a for a in angles if _angle_in_range(a, angle_lo, angle_hi)]
        # Ensure we have rays at the exact boundaries too
        angles.extend([angle_lo, angle_hi])

    # Deduplicate and sort
    angles = sorted(set(angles))

    if not angles:
        return []

    # ── 2. For each angle, find the nearest wall hit ────────────────────
    hits: List[Vec2] = []
    for angle in angles:
        rdx = math.cos(angle)
        rdy = math.sin(angle)

        closest_t = max_radius
        closest_pt: Optional[Vec2] = None

        for seg in wall_segments:
            t = _ray_segment_intersect(ox, oy, rdx, rdy, seg)
            if t is not None and 0 < t < closest_t:
                closest_t = t
                closest_pt = (ox + rdx * t, oy + rdy * t)

        if closest_pt is None:
            # No hit — use max_radius
            closest_pt = (ox + rdx * max_radius, oy + rdy * max_radius)

        hits.append(closest_pt)

    return hits


# ── Ray-segment intersection ────────────────────────────────────────────

def _ray_segment_intersect(
    ox: float, oy: float,
    rdx: float, rdy: float,
    seg: Seg,
) -> Optional[float]:
    """Compute the parameter *t* where the ray (ox+t*rdx, oy+t*rdy)
    intersects the line segment seg = ((sx1,sy1),(sx2,sy2)).

    Returns None if no valid intersection (parallel, behind origin,
    or outside segment bounds).
    """
    (sx1, sy1), (sx2, sy2) = seg
    sdx = sx2 - sx1
    sdy = sy2 - sy1

    denom = rdx * sdy - rdy * sdx
    if abs(denom) < 1e-12:
        return None  # parallel

    t = ((sx1 - ox) * sdy - (sy1 - oy) * sdx) / denom
    u = ((sx1 - ox) * rdy - (sy1 - oy) * rdx) / denom

    if t > 0 and 0.0 <= u <= 1.0:
        return t
    return None


# ── Angle helpers ───────────────────────────────────────────────────────

def _normalize_angle(a: float) -> float:
    """Normalise angle to [-π, π)."""
    while a < -math.pi:
        a += 2 * math.pi
    while a >= math.pi:
        a -= 2 * math.pi
    return a


def _angle_in_range(a: float, lo: float, hi: float) -> bool:
    """Check if angle *a* is within [lo, hi], handling wraparound."""
    a  = _normalize_angle(a)
    lo = _normalize_angle(lo)
    hi = _normalize_angle(hi)
    if lo <= hi:
        return lo <= a <= hi
    else:
        # Wraps around -π/π
        return a >= lo or a <= hi


# ── Utility: extract wall segments from a FloorPlan ─────────────────────

def extract_wall_segments_from_floorplan(floor_plan) -> List[Seg]:
    """Extract all wall segments from every room polygon in the floor plan.

    This gives us the complete set of interior + exterior walls that
    the isovist raycaster will intersect against.

    Parameters
    ----------
    floor_plan : FloorPlan

    Returns
    -------
    list of ((x1,y1),(x2,y2))
    """
    from door_placement.config import ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR

    segments: List[Seg] = []
    seen: set = set()

    for room in floor_plan.rooms:
        # Skip door pseudo-rooms — they are not walls
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            continue
        coords = list(room.poly.exterior.coords)
        for i in range(len(coords) - 1):
            p1 = (round(coords[i][0], 2), round(coords[i][1], 2))
            p2 = (round(coords[i+1][0], 2), round(coords[i+1][1], 2))
            # Deduplicate (order-independent)
            key = (min(p1, p2), max(p1, p2))
            if key not in seen:
                seen.add(key)
                segments.append((p1, p2))

    # Also add outer boundary segments
    if floor_plan.outer_boundary and not floor_plan.outer_boundary.is_empty:
        coords = list(floor_plan.outer_boundary.exterior.coords)
        for i in range(len(coords) - 1):
            p1 = (round(coords[i][0], 2), round(coords[i][1], 2))
            p2 = (round(coords[i+1][0], 2), round(coords[i+1][1], 2))
            key = (min(p1, p2), max(p1, p2))
            if key not in seen:
                seen.add(key)
                segments.append((p1, p2))

    return segments
