"""
Domain models for the door-placement engine.

Thin wrappers around Shapely geometries with semantic metadata attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, LineString, Point

from door_placement.config import (
    ROOM_TYPE_NAMES,
    ROOM_TYPE_LIVING,
    ROOM_TYPE_ENTRANCE,
    ROOM_TYPE_INT_DOOR,
)


@dataclass
class Room:
    """A single room parsed from the GSDiff JSON output."""

    room_id:   int
    type_id:   int
    poly:      Polygon
    name:      str = ""

    def __post_init__(self):
        if not self.name:
            self.name = ROOM_TYPE_NAMES.get(self.type_id, "Unknown")

    @property
    def area(self) -> float:
        return self.poly.area

    @property
    def centroid(self) -> Point:
        return self.poly.centroid


@dataclass
class WallSegment:
    """A straight wall segment extracted from room / building boundaries."""

    line:   LineString
    room_a: Optional[Room] = None   # room on one side
    room_b: Optional[Room] = None   # room on the other side

    @property
    def length(self) -> float:
        return self.line.length

    @property
    def p1(self) -> Tuple[float, float]:
        return self.line.coords[0]

    @property
    def p2(self) -> Tuple[float, float]:
        return self.line.coords[-1]

    @property
    def midpoint(self) -> Tuple[float, float]:
        x = (self.p1[0] + self.p2[0]) / 2
        y = (self.p1[1] + self.p2[1]) / 2
        return (x, y)

    def unit_tangent(self) -> Tuple[float, float]:
        """Unit vector along the wall."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def unit_normal(self) -> Tuple[float, float]:
        """Unit normal (perpendicular) to the wall — one of two possible."""
        ux, uy = self.unit_tangent()
        return (-uy, ux)

    def interpolate_at(self, distance: float) -> Tuple[float, float]:
        """Point on the segment at *distance* from p1."""
        pt = self.line.interpolate(distance)
        return (pt.x, pt.y)


@dataclass
class Door:
    """A placed door (entrance or internal)."""

    type_id:     int                # ROOM_TYPE_ENTRANCE or ROOM_TYPE_INT_DOOR
    poly:        Polygon            # the physical rectangle
    center:      Tuple[float, float] = (0.0, 0.0)
    normal:      Tuple[float, float] = (0.0, 0.0)
    connects:    Tuple[Optional[Room], Optional[Room]] = (None, None)
    score:       float = 0.0
    # Direction the door swings open into (unit vector).
    # Points from the hinge toward the room the door opens into.
    swing_direction: Tuple[float, float] = (0.0, 0.0)

    @property
    def name(self) -> str:
        return ROOM_TYPE_NAMES.get(self.type_id, "Door")


@dataclass
class FloorPlan:
    """Complete floor-plan state, accumulating placed elements."""

    rooms:          List[Room]            = field(default_factory=list)
    outer_boundary: Optional[Polygon]     = None
    doors:          List[Door]            = field(default_factory=list)
    resolution:     int                   = 512

    # Lazily computed helpers
    _main_living: Optional[Room]          = field(default=None, repr=False)
    _char_dim:    Optional[float]         = field(default=None, repr=False)

    # ── convenience accessors ───────────────────────────────────────────

    @property
    def main_living(self) -> Optional[Room]:
        """Largest living room (type 0) in the plan."""
        if self._main_living is None:
            livings = [r for r in self.rooms if r.type_id == ROOM_TYPE_LIVING]
            if livings:
                self._main_living = max(livings, key=lambda r: r.area)
        return self._main_living

    @property
    def characteristic_dimension(self) -> float:
        """Diagonal of the outer-boundary bounding box."""
        if self._char_dim is None:
            if self.outer_boundary and not self.outer_boundary.is_empty:
                b = self.outer_boundary.bounds
                self._char_dim = math.hypot(b[2] - b[0], b[3] - b[1])
            else:
                self._char_dim = float(self.resolution) * math.sqrt(2)
        return self._char_dim

    def rooms_by_type(self, type_id: int) -> List[Room]:
        return [r for r in self.rooms if r.type_id == type_id]

    def next_room_id(self) -> int:
        if not self.rooms:
            return 0
        return max(r.room_id for r in self.rooms) + 1

    def add_door(self, door: Door) -> None:
        """Register a door and also append it as a pseudo-room for JSON export."""
        self.doors.append(door)
        self.rooms.append(Room(
            room_id=self.next_room_id(),
            type_id=door.type_id,
            poly=door.poly,
            name=door.name,
        ))
