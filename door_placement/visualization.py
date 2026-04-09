"""
Visualisation helpers for floor plans with placed doors.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from door_placement.config import (
    VisualizationConfig,
    ROOM_TYPE_ENTRANCE,
    ROOM_TYPE_INT_DOOR,
)
from door_placement.models import FloorPlan


def shapely_to_pil(geom: Polygon) -> list:
    """Convert a Shapely polygon to a flat list of (x, y) tuples for PIL."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        # Pick the largest polygon
        parts = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not parts:
            return []
        geom = max(parts, key=lambda p: p.area)
    return [(int(p[0]), int(p[1])) for p in geom.exterior.coords]


def render_floor_plan(
    floor_plan: FloorPlan,
    config: Optional[VisualizationConfig] = None,
    show_cone: bool = False,
    cone_polygon: Optional[Polygon] = None,
) -> Image.Image:
    """Render the complete floor plan to a PIL Image.

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

    # ── 1. Draw regular rooms (not doors) ───────────────────────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            continue
        color = config.room_colors.get(room.type_id, (220, 220, 220))
        coords = shapely_to_pil(room.poly)
        if coords:
            draw.polygon(coords, fill=color, outline=config.wall_color)

    # ── 2. Draw outer boundary ──────────────────────────────────────────
    if floor_plan.outer_boundary and not floor_plan.outer_boundary.is_empty:
        outer_coords = shapely_to_pil(floor_plan.outer_boundary)
        if outer_coords:
            draw.polygon(outer_coords,
                         outline=config.outline_color,
                         width=config.outline_width)

    # ── 3. Draw vision cone (optional) ──────────────────────────────────
    if show_cone and cone_polygon and not cone_polygon.is_empty:
        cone_coords = shapely_to_pil(cone_polygon)
        if cone_coords:
            draw.polygon(cone_coords,
                         fill=config.cone_fill,
                         outline=config.cone_outline)

    # ── 4. Draw doors on top ────────────────────────────────────────────
    for room in floor_plan.rooms:
        if room.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            color = config.room_colors.get(room.type_id, (0, 0, 0))
            coords = shapely_to_pil(room.poly)
            if coords:
                draw.polygon(coords, fill=color,
                             outline=config.outline_color, width=2)

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
