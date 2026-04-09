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


# ── Bed-position estimation ─────────────────────────────────────────────

def estimate_bed_position(
    room_poly: Polygon,
    door_center: Tuple[float, float],
    bed_size_ratio: float = 0.25,
) -> Polygon:
    """Estimate a likely bed rectangle inside a bedroom.

    Heuristic: the bed is placed against the wall segment that is
    **furthest** from the door.  The bed occupies ``bed_size_ratio``
    of the room's shortest bounding-box dimension as its short side,
    and twice that as its long side.

    Parameters
    ----------
    room_poly : Polygon
        The bedroom polygon.
    door_center : (x, y)
        Centre of the placed (or candidate) door.
    bed_size_ratio : float
        Fraction of the room's shortest dimension used as bed width.

    Returns
    -------
    Polygon
        Estimated bed rectangle.
    """
    bounds = room_poly.bounds  # (minx, miny, maxx, maxy)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    short_dim = min(w, h)

    bed_short = short_dim * bed_size_ratio
    bed_long  = bed_short * 2.0

    # Find the wall segment furthest from the door
    coords = list(room_poly.exterior.coords)
    best_seg = None
    best_dist = -1.0
    door_pt = Point(door_center)

    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        if seg.length < bed_long * 0.5:
            continue  # too short to place a bed
        d = seg.distance(door_pt)
        if d > best_dist:
            best_dist = d
            best_seg = seg

    if best_seg is None:
        # Fallback: use room centroid
        cx, cy = room_poly.centroid.x, room_poly.centroid.y
        return Polygon([
            (cx - bed_long/2, cy - bed_short/2),
            (cx + bed_long/2, cy - bed_short/2),
            (cx + bed_long/2, cy + bed_short/2),
            (cx - bed_long/2, cy + bed_short/2),
        ])

    # Place bed at the midpoint of the furthest wall
    mid = best_seg.interpolate(0.5, normalized=True)
    mx, my = mid.x, mid.y

    p1, p2 = best_seg.coords[0], best_seg.coords[-1]
    (ux, uy), (nx, ny) = segment_tangent_normal(p1, p2)

    # The bed lies flat against the wall:
    #   long side along the wall tangent, short side into the room
    # We need the inward normal to push the bed into the room
    probe_in = Point(mx + nx * 5, my + ny * 5)
    if not room_poly.contains(probe_in):
        nx, ny = -nx, -ny  # flip normal

    corners = [
        (mx - bed_long/2 * ux,             my - bed_long/2 * uy),
        (mx + bed_long/2 * ux,             my + bed_long/2 * uy),
        (mx + bed_long/2 * ux + bed_short * nx, my + bed_long/2 * uy + bed_short * ny),
        (mx - bed_long/2 * ux + bed_short * nx, my - bed_long/2 * uy + bed_short * ny),
    ]
    bed = Polygon(corners)

    # Clip to room interior
    bed = bed.intersection(room_poly)
    if bed.is_empty:
        cx, cy = room_poly.centroid.x, room_poly.centroid.y
        return Polygon([
            (cx - bed_long/2, cy - bed_short/2),
            (cx + bed_long/2, cy - bed_short/2),
            (cx + bed_long/2, cy + bed_short/2),
            (cx - bed_long/2, cy + bed_short/2),
        ])

    return bed


# ── Sightline depth measurement ─────────────────────────────────────────

def measure_sightline_depth(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    room_poly: Polygon,
) -> float:
    """Measure how far a straight sightline penetrates into a room.

    Casts a single ray from *origin* in *direction* and returns the
    distance to the first intersection with the room boundary.

    Parameters
    ----------
    origin : (x, y)
    direction : (dx, dy)   unit vector
    room_poly : Polygon

    Returns
    -------
    float
        Distance in pixels.  0.0 if no intersection.
    """
    ox, oy = origin
    dx, dy = direction
    length = math.hypot(room_poly.bounds[2] - room_poly.bounds[0],
                        room_poly.bounds[3] - room_poly.bounds[1]) * 2

    far_pt = (ox + dx * length, oy + dy * length)
    ray = LineString([(ox, oy), far_pt])

    intersection = ray.intersection(room_poly.exterior)
    if intersection.is_empty:
        return 0.0

    if intersection.geom_type == "Point":
        return Point(origin).distance(intersection)
    elif intersection.geom_type == "MultiPoint":
        return min(Point(origin).distance(pt) for pt in intersection.geoms)
    else:
        return Point(origin).distance(intersection)
