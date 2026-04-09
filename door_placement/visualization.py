"""
Visualisation helpers for floor plans with placed doors.

The rendering style exactly matches the GSDiff model output:
  - Rooms are filled with their semantic colour (no outline on fill).
  - Wall edges are drawn as thick gray lines on top of the fills.
  - Corner dots are drawn at each vertex as gray squares.
  - Doors are drawn as coloured rectangles with a swing-arc overlay
    showing which direction the door opens.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from door_placement.config import (
    VisualizationConfig,
    ROOM_TYPE_ENTRANCE,
    ROOM_TYPE_INT_DOOR,
)
from door_placement.models import FloorPlan


# ── Helpers ─────────────────────────────────────────────────────────────

def shapely_to_pil(geom: Polygon) -> list:
    """Convert a Shapely polygon to a flat list of (x, y) tuples for PIL."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not parts:
            return []
        geom = max(parts, key=lambda p: p.area)
    return [(int(p[0]), int(p[1])) for p in geom.exterior.coords]


def _polygon_vertices(geom: Polygon) -> List[Tuple[int, int]]:
    """Return vertices without the closing duplicate."""
    coords = shapely_to_pil(geom)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def _draw_swing_arc(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    swing_dx: float, swing_dy: float,
    radius: float,
    color: tuple = (0, 180, 0),
    arc_degrees: float = 90.0,
    width: int = 2,
) -> None:
    """Draw a door-swing arc on the image.

    The arc starts from the swing_direction vector and sweeps
    ``arc_degrees`` counter-clockwise.

    Parameters
    ----------
    draw : ImageDraw
    cx, cy : float
        Centre (hinge) position.
    swing_dx, swing_dy : float
        Unit vector of the initial swing direction.
    radius : float
        Arc radius (≈ door width).
    color : tuple
        RGB colour of the arc.
    arc_degrees : float
        Sweep angle in degrees.
    width : int
        Line width.
    """
    # PIL arcs use degrees measured counter-clockwise from the positive X-axis.
    start_angle_deg = math.degrees(math.atan2(-swing_dy, swing_dx))  # PIL Y is inverted
    end_angle_deg = start_angle_deg + arc_degrees

    bbox = [
        cx - radius, cy - radius,
        cx + radius, cy + radius,
    ]
    draw.arc(bbox, start=start_angle_deg, end=end_angle_deg,
             fill=color, width=width)

    # Draw a small line from centre to arc start (hinge indicator)
    end_x = cx + swing_dx * radius
    end_y = cy + swing_dy * radius
    draw.line([(int(cx), int(cy)), (int(end_x), int(end_y))],
              fill=color, width=width)


# ── Main renderer ───────────────────────────────────────────────────────

def render_floor_plan(
    floor_plan: FloorPlan,
    config: Optional[VisualizationConfig] = None,
    show_cone: bool = False,
    cone_polygon: Optional[Polygon] = None,
) -> Image.Image:
    """Render the complete floor plan to a PIL Image.

    The drawing style matches the GSDiff model's output:
      1. White background
      2. Fill each room polygon with its semantic colour (no outline)
      3. Draw gray wall-lines on top of room edges
      4. Draw gray corner-dots at each vertex
      5. Draw doors as coloured rectangles
      6. Draw door-swing arcs

    Parameters
    ----------
    floor_plan : FloorPlan
    config : VisualizationConfig, optional
    show_cone : bool
        If True, draw the entrance vision cone.
    cone_polygon : Polygon, optional
        The cone to draw (only used when *show_cone* is True).

    Returns
    -------
    PIL.Image.Image
    """
    if config is None:
        config = VisualizationConfig()

    res = floor_plan.resolution
    img = Image.new("RGB", (res, res), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    wall_color = config.wall_color       # (150, 150, 150)
    wall_width = 7                        # matches GSDiff model output
    corner_rad = 3                        # radius of corner dots

    # ── 1. Fill rooms with semantic colour (no outline) ──────────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            continue
        color = config.room_colors.get(room.type_id, (220, 220, 220))
        coords = shapely_to_pil(room.poly)
        if coords:
            draw.polygon(coords, fill=color, outline=None)

    # ── 2. Draw wall edges (thick gray lines on top of fills) ────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            continue
        verts = _polygon_vertices(room.poly)
        if not verts:
            continue
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            draw.line([p1, p2], fill=wall_color, width=wall_width)

    # ── 3. Draw corner dots (gray squares at each vertex) ────────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            continue
        verts = _polygon_vertices(room.poly)
        for v in verts:
            x, y = v
            draw.rectangle(
                [x - corner_rad, y - corner_rad,
                 x + corner_rad, y + corner_rad],
                fill=wall_color, outline=None
            )

    # ── 4. Draw outer boundary (thick black outline) ─────────────────────
    if floor_plan.outer_boundary and not floor_plan.outer_boundary.is_empty:
        outer_coords = shapely_to_pil(floor_plan.outer_boundary)
        if outer_coords:
            draw.polygon(outer_coords,
                         outline=config.outline_color,
                         width=config.outline_width)

    # ── 5. Draw vision cone (optional) ──────────────────────────────────
    if show_cone and cone_polygon and not cone_polygon.is_empty:
        cone_coords = shapely_to_pil(cone_polygon)
        if cone_coords:
            draw.polygon(cone_coords,
                         fill=config.cone_fill,
                         outline=config.cone_outline)

    # ── 6. Draw doors as coloured rectangles ────────────────────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            color = config.room_colors.get(room.type_id, (0, 0, 0))
            coords = shapely_to_pil(room.poly)
            if coords:
                draw.polygon(coords, fill=color,
                             outline=config.outline_color, width=2)

    # ── 7. Draw door-swing arcs ─────────────────────────────────────────
    for door in floor_plan.doors:
        sx, sy = door.swing_direction
        if sx == 0.0 and sy == 0.0:
            continue  # no swing data

        cx, cy = door.center

        # Arc radius = approximate half the door width (from polygon)
        bounds = door.poly.bounds  # (minx, miny, maxx, maxy)
        dx = bounds[2] - bounds[0]
        dy = bounds[3] - bounds[1]
        radius = max(dx, dy) * 0.5

        arc_color = (0, 180, 0) if door.type_id == ROOM_TYPE_INT_DOOR else (0, 150, 255)
        _draw_swing_arc(draw, cx, cy, sx, sy, radius,
                        color=arc_color, arc_degrees=90, width=2)

    return img


def save_visualization(
    floor_plan: FloorPlan,
    output_path: str,
    config: Optional[VisualizationConfig] = None,
    **kwargs,
) -> None:
    """Render and save the floor plan to a PNG file.

    Parameters
    ----------
    floor_plan : FloorPlan
    output_path : str
    config : VisualizationConfig, optional
    **kwargs
        Forwarded to :func:`render_floor_plan`.
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img = render_floor_plan(floor_plan, config, **kwargs)
    img.save(output_path)
    print(f"[*] Visualization saved to: {output_path}")
