"""
Load / save GSDiff-format floor plan JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from shapely.geometry import Polygon

from door_placement.models import Room, FloorPlan
from door_placement.config import ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR


def load_floorplan(path: Union[str, Path], resolution: int = 512) -> FloorPlan:
    """Parse a GSDiff JSON file into a :class:`FloorPlan`.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file produced by the GSDiff pipeline.
    resolution : int
        Image resolution used during generation (default 512).

    Returns
    -------
    FloorPlan
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    rooms = []
    for r in data.get("rooms", []):
        coords = r.get("coordinates", [])
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid or poly.is_empty:
            continue
        rooms.append(Room(
            room_id=r["room_id"],
            type_id=r["room_type_id"],
            poly=poly,
            name=r.get("room_type_name", ""),
        ))

    outer_coords = data.get("outer_boundary", [])
    outer_poly = Polygon(outer_coords) if len(outer_coords) >= 3 else None

    return FloorPlan(
        rooms=rooms,
        outer_boundary=outer_poly,
        resolution=resolution,
    )


def save_floorplan(fp: FloorPlan, path: Union[str, Path]) -> None:
    """Serialise a :class:`FloorPlan` back to GSDiff-compatible JSON.

    For door rooms (entrance / internal), the output also includes
    ``swing_direction`` — a unit-vector [dx, dy] showing which way
    the door swings open.

    Parameters
    ----------
    fp : FloorPlan
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rooms_data = []
    for r in fp.rooms:
        coords = [[int(x), int(y)] for x, y in r.poly.exterior.coords]
        room_dict = {
            "room_id":       r.room_id,
            "room_type_id":  r.type_id,
            "room_type_name": r.name,
            "coordinates":   coords,
        }

        # For doors, also export swing direction from the Door object
        if r.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR):
            # Find the matching Door object
            matching_door = None
            for d in fp.doors:
                if d.type_id == r.type_id and d.poly.equals(r.poly):
                    matching_door = d
                    break
            if matching_door is not None:
                sx, sy = matching_door.swing_direction
                room_dict["swing_direction"] = [round(sx, 4), round(sy, 4)]
                room_dict["center"] = [
                    round(matching_door.center[0], 2),
                    round(matching_door.center[1], 2),
                ]

        rooms_data.append(room_dict)

    outer_coords = []
    if fp.outer_boundary and not fp.outer_boundary.is_empty:
        outer_coords = [[int(x), int(y)]
                        for x, y in fp.outer_boundary.exterior.coords]

    data = {
        "rooms":         rooms_data,
        "outer_boundary": outer_coords,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

