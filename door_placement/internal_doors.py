"""
Internal door placement algorithm.

Places doors between adjacent rooms using topological adjacency
detection and smart wall-offset positioning.

This module should be executed **before** the entrance-door algorithm
because internal doors are more geometrically constrained (they must
sit on a shared wall), and the entrance scoring benefits from knowing
where internal doors are.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from door_placement.config import (
    InternalDoorConfig,
    ROOM_TYPE_LIVING,
    ROOM_TYPE_BEDROOM,
    ROOM_TYPE_KITCHEN,
    ROOM_TYPE_STORAGE,
    ROOM_TYPE_BATHROOM,
    ROOM_TYPE_ENTRANCE,
    ROOM_TYPE_INT_DOOR,
    ROOM_TYPE_STAIRS,
)
from door_placement.models import Room, Door, FloorPlan, WallSegment
from door_placement.geometry_utils import (
    extract_segments,
    segment_tangent_normal,
    build_door_polygon,
)


# ── Public API ──────────────────────────────────────────────────────────

def place_internal_doors(
    floor_plan: FloorPlan,
    config: Optional[InternalDoorConfig] = None,
) -> List[Door]:
    """Place internal doors for every room that needs one.

    The algorithm:
        1. For each non-living, non-entrance room, find adjacent rooms.
        2. Pick the best target (prefer living rooms, then longest wall).
        3. Find the longest shared wall segment.
        4. Position the door at an offset from the nearer corner
           (not the midpoint) to maximise usable wall space.
        5. Check for collisions with previously placed doors.
        6. Register the door on the floor plan.

    Parameters
    ----------
    floor_plan : FloorPlan
        Must already contain rooms with valid polygons.
    config : InternalDoorConfig, optional

    Returns
    -------
    list[Door]
        All successfully placed internal doors.
    """
    if config is None:
        config = InternalDoorConfig()

    cd = floor_plan.characteristic_dimension
    placed_doors: List[Door] = []
    min_spacing = cd * config.min_door_spacing_ratio

    # Rooms that *generate* doors (not living rooms, entrances, etc.)
    door_generating_types = {
        ROOM_TYPE_BEDROOM,
        ROOM_TYPE_KITCHEN,
        ROOM_TYPE_STORAGE,
        ROOM_TYPE_BATHROOM,
    }

    for current_room in floor_plan.rooms:
        if current_room.type_id not in door_generating_types:
            continue

        # ── Step 1: find adjacent rooms ─────────────────────────────────
        adjacents = _find_adjacent_rooms(
            current_room, floor_plan.rooms, config
        )
        if not adjacents:
            print(f"[!] Room {current_room.room_id} ({current_room.name}) "
                  f"is completely isolated — skipping.")
            continue

        # ── Step 2: pick best target based on room type ─────────────────
        target_info = _pick_target_room(
            current_room, adjacents, config
        )
        if target_info is None:
            continue

        target_room, shared_geom = target_info

        # ── Step 3: get the door width for this room type ───────────────
        width_ratio = config.width_ratios.get(
            current_room.type_id, config.default_width_ratio
        )
        if width_ratio == 0.0:
            # Room type has no physical door (e.g. kitchen archway)
            print(f"[~] Room {current_room.room_id} ({current_room.name}) "
                  f"gets an open archway — no door placed.")
            continue

        door_width = cd * width_ratio
        door_depth = cd * config.default_depth_ratio

        # ── Step 4: find & choose the best shared wall segment ──────────
        segments = extract_segments(shared_geom.simplify(2.0),
                                     min_length=door_width)
        if not segments:
            # Fallback: try without minimum-length filter
            segments = extract_segments(shared_geom.simplify(2.0))
        if not segments:
            print(f"[!] No usable wall between Room {current_room.room_id} "
                  f"and Room {target_room.room_id} — skipping.")
            continue

        best_seg = max(segments, key=lambda s: s.length)

        # ── Step 5: smart offset positioning ────────────────────────────
        door_center = _compute_door_position(
            best_seg, door_width, config.offset_from_corner_ratio
        )

        # ── Step 6: build the door polygon ──────────────────────────────
        p1, p2 = best_seg.coords[0], best_seg.coords[-1]
        (ux, uy), (nx, ny) = segment_tangent_normal(p1, p2)

        door_poly = build_door_polygon(
            cx=door_center[0],
            cy=door_center[1],
            ux=ux, uy=uy,
            nx=nx, ny=ny,
            half_width=door_width / 2,
            half_depth=door_depth / 2,
        )

        # ── Step 7: collision check with existing doors ─────────────────
        if _collides_with_existing(door_poly, placed_doors, min_spacing):
            # Try the mirror position (other end of the wall)
            alt_center = _compute_door_position(
                best_seg, door_width,
                1.0 - config.offset_from_corner_ratio,
            )
            door_poly = build_door_polygon(
                cx=alt_center[0], cy=alt_center[1],
                ux=ux, uy=uy, nx=nx, ny=ny,
                half_width=door_width / 2, half_depth=door_depth / 2,
            )
            if _collides_with_existing(door_poly, placed_doors, min_spacing):
                # Last resort: midpoint
                mid = best_seg.interpolate(0.5, normalized=True)
                door_poly = build_door_polygon(
                    cx=mid.x, cy=mid.y,
                    ux=ux, uy=uy, nx=nx, ny=ny,
                    half_width=door_width / 2, half_depth=door_depth / 2,
                )
                if _collides_with_existing(door_poly, placed_doors, min_spacing):
                    print(f"[!] Cannot place door for Room "
                          f"{current_room.room_id} — all positions collide.")
                    continue

        # ── Step 8: register ────────────────────────────────────────────
        door = Door(
            type_id=ROOM_TYPE_INT_DOOR,
            poly=door_poly,
            center=(door_poly.centroid.x, door_poly.centroid.y),
            normal=(nx, ny),
            connects=(current_room, target_room),
        )
        placed_doors.append(door)
        floor_plan.add_door(door)
        print(f"[+] Door placed: {current_room.name} → {target_room.name}")

    return placed_doors


# ── Private helpers ─────────────────────────────────────────────────────

def _find_adjacent_rooms(
    room: Room,
    all_rooms: List[Room],
    config: InternalDoorConfig,
) -> List[Tuple[Room, Polygon]]:
    """Return rooms that share a wall (or near-wall) with *room*."""
    buf = config.adjacency_buffer_px
    threshold = config.adjacency_area_threshold
    result = []

    for other in all_rooms:
        if other.room_id == room.room_id:
            continue
        # Skip doors / stairs / entrance pseudo-rooms
        if other.type_id in (ROOM_TYPE_ENTRANCE, ROOM_TYPE_INT_DOOR,
                             ROOM_TYPE_STAIRS):
            continue

        shared = room.poly.buffer(buf).intersection(other.poly.buffer(buf))
        if not shared.is_empty and shared.area > threshold:
            result.append((other, shared))

    return result


def _pick_target_room(
    current_room: Room,
    adjacents: List[Tuple[Room, Polygon]],
    config: InternalDoorConfig,
) -> Optional[Tuple[Room, Polygon]]:
    """Choose which adjacent room to connect *current_room* to.

    Priority:
        • Bedrooms / Kitchens / Storage → prefer Living Room
        • Bathrooms → prefer Living (hall bath) or Bedroom (en-suite)
        • Fallback → whichever shares the longest wall
    """
    rtype = current_room.type_id

    # ── Bedroom, Kitchen, Storage: try living room first ────────────────
    if rtype in (ROOM_TYPE_BEDROOM, ROOM_TYPE_KITCHEN, ROOM_TYPE_STORAGE):
        living_adj = [(r, g) for r, g in adjacents
                      if r.type_id == ROOM_TYPE_LIVING]
        if living_adj:
            return max(living_adj, key=lambda x: x[1].length)
        return max(adjacents, key=lambda x: x[1].length)

    # ── Bathroom: hall bath vs en-suite ──────────────────────────────────
    if rtype == ROOM_TYPE_BATHROOM:
        living_adj = [(r, g) for r, g in adjacents
                      if r.type_id == ROOM_TYPE_LIVING]
        bed_adj = [(r, g) for r, g in adjacents
                   if r.type_id == ROOM_TYPE_BEDROOM]

        # If it has a decent wall with a living room → hall bath
        valid_living = [(r, g) for r, g in living_adj
                        if g.length > floor_plan_door_width_hint(config)]
        if valid_living:
            return max(valid_living, key=lambda x: x[1].length)
        # Otherwise connect to bedroom → en-suite
        if bed_adj:
            return max(bed_adj, key=lambda x: x[1].length)
        return max(adjacents, key=lambda x: x[1].length)

    # Fallback for any other type
    return max(adjacents, key=lambda x: x[1].length)


def floor_plan_door_width_hint(config: InternalDoorConfig) -> float:
    """Quick estimate of door width for the 'valid living wall' test.

    We use a rough approximation since we don't have char_dim here;
    the test is just to distinguish hall-bath from en-suite.
    """
    return 30.0  # sensible pixel-estimate for adjacency check


def _compute_door_position(
    segment: object,  # LineString
    door_width: float,
    offset_ratio: float,
) -> Tuple[float, float]:
    """Compute door centre position along a wall segment.

    Instead of always placing at the midpoint, we offset the door
    ``offset_ratio`` of the way along the wall from the nearer end,
    clamping so the door rectangle doesn't hang over either end.

    Parameters
    ----------
    segment : LineString
    door_width : float
    offset_ratio : float
        0.3 means 30 % from one end.

    Returns
    -------
    (x, y) : tuple[float, float]
    """
    seg_len = segment.length
    half_door = door_width / 2

    # Clamp: keep at least half a door-width from each end
    min_dist = half_door
    max_dist = seg_len - half_door

    if max_dist <= min_dist:
        # Wall is barely long enough — just use midpoint
        dist = seg_len / 2
    else:
        dist = min_dist + (max_dist - min_dist) * offset_ratio

    pt = segment.interpolate(dist)
    return (pt.x, pt.y)


def _collides_with_existing(
    new_poly: Polygon,
    existing_doors: List[Door],
    min_spacing: float,
) -> bool:
    """Return True if *new_poly* is too close to any existing door."""
    for d in existing_doors:
        if new_poly.distance(d.poly) < min_spacing:
            return True
    return False
