"""
Shared geometry utilities used by both door-placement algorithms.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

from shapely.geometry import Polygon, LineString, Point


# ── Line extraction ─────────────────────────────────────────────────────

def extract_segments(geom, min_length: float = 0.0) -> List[LineString]:
    """Recursively extract individual straight-line segments from any
    Shapely geometry (LineString, MultiLineString, Polygon, Collection).

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
    min_length : float
        Discard segments shorter than this.

    Returns
    -------
    list[LineString]
    """
    segments: List[LineString] = []
    _extract(geom, segments)
    if min_length > 0:
        segments = [s for s in segments if s.length >= min_length]
    return segments


def _extract(geom, out: list) -> None:
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "LineString":
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            out.append(LineString([coords[i], coords[i + 1]]))
    elif gtype == "Polygon":
        coords = list(geom.exterior.coords)
        for i in range(len(coords) - 1):
            out.append(LineString([coords[i], coords[i + 1]]))
    elif gtype in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            _extract(g, out)


# ── Wall / segment helpers ──────────────────────────────────────────────

def segment_tangent_normal(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (unit_tangent, unit_normal) for the segment p1→p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0), (0.0, 1.0)
    ux, uy = dx / length, dy / length
    return (ux, uy), (-uy, ux)


def inward_normal(
    segment: LineString,
    reference_poly: Polygon,
    probe_distance: float = 5.0,
) -> Tuple[float, float]:
    """Return the unit normal that points *into* ``reference_poly``.

    We test both candidate normals; whichever probe-point falls inside
    the polygon is the inward one.
    """
    p1, p2 = segment.coords[0], segment.coords[-1]
    (ux, uy), (nx, ny) = segment_tangent_normal(p1, p2)
    mid = segment.interpolate(0.5, normalized=True)
    mx, my = mid.x, mid.y

    if reference_poly.contains(Point(mx + nx * probe_distance,
                                     my + ny * probe_distance)):
        return (nx, ny)
    return (-nx, -ny)


# ── Vision-cone construction ────────────────────────────────────────────

def build_vision_cone(
    px: float,
    py: float,
    nx: float,
    ny: float,
    spread_deg: float = 30.0,
    length: float = 200.0,
) -> Polygon:
    """Create a triangular vision-cone polygon.

    Parameters
    ----------
    px, py : float
        Origin of the cone (door position).
    nx, ny : float
        Unit inward-normal direction.
    spread_deg : float
        Half-angle of the cone in degrees.
    length : float
        How far the cone extends.

    Returns
    -------
    Polygon
        Triangle {origin, left_bound, right_bound}.
    """
    rad = math.radians(spread_deg)

    # Rotate the normal by ±spread to get cone edges
    cos_l, sin_l = math.cos(-rad), math.sin(-rad)
    lx = px + (nx * cos_l - ny * sin_l) * length
    ly = py + (nx * sin_l + ny * cos_l) * length

    cos_r, sin_r = math.cos(rad), math.sin(rad)
    rx = px + (nx * cos_r - ny * sin_r) * length
    ry = py + (nx * sin_r + ny * cos_r) * length

    return Polygon([(px, py), (lx, ly), (rx, ry)])


def clip_cone_to_walls(
    cone: Polygon,
    room_polygons: List[Polygon],
) -> Polygon:
    """Clip a vision cone so it stops at the first wall it hits.

    This prevents the cone from "seeing through" internal walls,
    producing a more realistic visibility test.

    For efficiency we only intersect with the target living room;
    a full raycast could be added later.
    """
    clipped = cone
    for poly in room_polygons:
        # Subtract everything *outside* the room from the cone
        clipped = clipped.intersection(poly)
        if clipped.is_empty:
            break
    return clipped


# ── Door-polygon construction ───────────────────────────────────────────

def build_door_polygon(
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    nx: float,
    ny: float,
    half_width: float,
    half_depth: float,
) -> Polygon:
    """Construct an axis-aligned door rectangle centred at (cx, cy).

    The rectangle is oriented so that "width" runs along the wall
    (tangent direction ux,uy) and "depth" runs perpendicular to
    it (normal direction nx,ny).

    Returns
    -------
    Polygon
        Four-corner rectangle.
    """
    corners = [
        (cx - half_width * ux - half_depth * nx,
         cy - half_width * uy - half_depth * ny),
        (cx + half_width * ux - half_depth * nx,
         cy + half_width * uy - half_depth * ny),
        (cx + half_width * ux + half_depth * nx,
         cy + half_width * uy + half_depth * ny),
        (cx - half_width * ux + half_depth * nx,
         cy - half_width * uy + half_depth * ny),
    ]
    return Polygon(corners)


# ── Door-swing arc validation ───────────────────────────────────────────

def validate_door_swing(
    hinge: Tuple[float, float],
    door_width: float,
    swing_direction: Tuple[float, float],
    obstacles: List[Polygon],
    arc_degrees: float = 90.0,
    arc_resolution: int = 16,
) -> bool:
    """Check whether a door can swing open without hitting any obstacle.

    We approximate the swing arc as a pie-slice polygon and test
    intersection against each obstacle.

    Parameters
    ----------
    hinge : (x, y)
    door_width : float
        Radius of the swing arc.
    swing_direction : (dx, dy)
        Unit vector in the *initial* closed-door direction.
    obstacles : list[Polygon]
        Walls, other doors, etc.
    arc_degrees : float
        How far the door swings (default 90°).
    arc_resolution : int
        Number of points used to approximate the arc.

    Returns
    -------
    bool
        ``True`` if the swing is clear.
    """
    hx, hy = hinge
    sdx, sdy = swing_direction

    # Build arc polygon
    start_angle = math.atan2(sdy, sdx)
    end_angle = start_angle + math.radians(arc_degrees)

    points = [(hx, hy)]
    for i in range(arc_resolution + 1):
        a = start_angle + (end_angle - start_angle) * i / arc_resolution
        points.append((hx + door_width * math.cos(a),
                        hy + door_width * math.sin(a)))
    arc_poly = Polygon(points)

    for obs in obstacles:
        if arc_poly.intersects(obs):
            return False
    return True
