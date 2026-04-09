"""
Internal door placement algorithm with isovist-based privacy scoring.

Places doors between adjacent rooms using topological adjacency
detection and multi-criteria privacy optimisation via isovist
(visibility polygon) analysis.

This module should be executed **before** the entrance-door algorithm
because internal doors are more geometrically constrained (they must
sit on a shared wall), and the entrance scoring benefits from knowing
where internal doors are.

Scoring criteria (when isovist scoring is enabled):
    1. Privacy       — minimise private-area exposure from public side
    2. Bed concealment — bed NOT visible from entry gaze
    3. Transition zone — prefer positions that create a natural turn
    4. Furniture      — maximise usable continuous wall length
    5. Public exposure — penalise entrance sightlines into bedrooms
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
    inward_normal,
    build_door_polygon,
    estimate_bed_position,
    measure_sightline_depth,
)
from door_placement.isovist import (
    compute_isovist,
    extract_wall_segments_from_floorplan,
)


# ── Public API ──────────────────────────────────────────────────────────

def place_internal_doors(
    floor_plan: FloorPlan,
    config: Optional[InternalDoorConfig] = None,
) -> List[Door]:
    """Place internal doors for every room that needs one.

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

    # Precompute shared data for isovist scoring
    private_polys = [
        r.poly for r in floor_plan.rooms
        if r.type_id == ROOM_TYPE_BEDROOM
    ]
    unified_private = unary_union(private_polys) if private_polys else Polygon()

    wall_segments = None
    if config.enable_isovist_scoring:
        wall_segments = extract_wall_segments_from_floorplan(floor_plan)

    # Rooms that *generate* doors
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
            print(f"[~] Room {current_room.room_id} ({current_room.name}) "
                  f"gets an open archway — no door placed.")
            continue

        door_width = cd * width_ratio
        door_depth = cd * config.default_depth_ratio

        # ── Step 4: find shared wall segments ───────────────────────────
        segments = extract_segments(shared_geom.simplify(2.0),
                                     min_length=door_width)
        if not segments:
            segments = extract_segments(shared_geom.simplify(2.0))
        if not segments:
            print(f"[!] No usable wall between Room {current_room.room_id} "
                  f"and Room {target_room.room_id} — skipping.")
            continue

        best_seg = max(segments, key=lambda s: s.length)

        # ── Step 5: position the door ───────────────────────────────────
        if (config.enable_isovist_scoring
                and wall_segments is not None
                and not unified_private.is_empty):
            door_center, door_poly, best_score = _isovist_score_positions(
                best_seg, door_width, door_depth,
                current_room, target_room,
                unified_private,
                placed_doors, min_spacing,
                cd, config,
                wall_segments, floor_plan,
            )
        else:
            # Fallback: corner-offset positioning
            door_center = _compute_door_position(
                best_seg, door_width, config.offset_from_corner_ratio
            )
            p1, p2 = best_seg.coords[0], best_seg.coords[-1]
            (ux, uy), (nx, ny) = segment_tangent_normal(p1, p2)
            door_poly = build_door_polygon(
                cx=door_center[0], cy=door_center[1],
                ux=ux, uy=uy, nx=nx, ny=ny,
                half_width=door_width / 2, half_depth=door_depth / 2,
            )
            best_score = 0.0

        if door_poly is None:
            print(f"[!] Cannot place door for Room "
                  f"{current_room.room_id} — no valid position found.")
            continue

        # ── Step 6: collision fallback ──────────────────────────────────
        if _collides_with_existing(door_poly, placed_doors, min_spacing):
            p1, p2 = best_seg.coords[0], best_seg.coords[-1]
            (ux, uy), (nx, ny) = segment_tangent_normal(p1, p2)
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

        # ── Step 7: register ────────────────────────────────────────────
        p1, p2 = best_seg.coords[0], best_seg.coords[-1]
        (_, _), (nx, ny) = segment_tangent_normal(p1, p2)
        swing_dir = inward_normal(best_seg, target_room.poly)

        door = Door(
            type_id=ROOM_TYPE_INT_DOOR,
            poly=door_poly,
            center=(door_poly.centroid.x, door_poly.centroid.y),
            normal=(nx, ny),
            connects=(current_room, target_room),
            swing_direction=swing_dir,
            score=best_score,
        )
        placed_doors.append(door)
        floor_plan.add_door(door)
        print(f"[+] Door placed: {current_room.name} → {target_room.name} "
              f"(privacy score: {best_score:.2f})")

    return placed_doors


# ── Isovist-based multi-criteria scorer ─────────────────────────────────

def _isovist_score_positions(
    segment,             # LineString — shared wall
    door_width: float,
    door_depth: float,
    current_room: Room,
    target_room: Room,
    unified_private: Polygon,
    placed_doors: List[Door],
    min_spacing: float,
    char_dim: float,
    config: InternalDoorConfig,
    wall_segments: list,
    floor_plan: FloorPlan,
) -> Tuple[Optional[Tuple[float, float]], Optional[Polygon], float]:
    """Score every candidate position using isovist-based multi-criteria.

    For each candidate position along the shared wall:
        1. Build the door polygon; skip if collision.
        2. Compute two probe points (public side, private side).
        3. Compute isovist from the public probe point.
        4. Score on 5 criteria.
        5. Return the position with the highest total score.
    """
    seg_len = segment.length
    p1_coords = segment.coords[0]
    p2_coords = segment.coords[-1]
    (ux, uy), (nx, ny) = segment_tangent_normal(p1_coords, p2_coords)

    # Determine inward normals
    in_to_target = inward_normal(segment, target_room.poly)
    in_to_current = (-in_to_target[0], -in_to_target[1])

    isovist_radius = char_dim * config.isovist_max_radius_ratio
    step = config.isovist_slide_step_px
    half_door = door_width / 2
    probe_offset = config.probe_offset

    usable_start = half_door
    usable_end = seg_len - half_door
    if usable_end <= usable_start:
        usable_start = seg_len / 2
        usable_end = seg_len / 2

    num_steps = max(1, int((usable_end - usable_start) / step))

    # Precompute private area for normalisation
    private_area = unified_private.area
    if private_area < 1.0:
        private_area = 1.0

    # Precompute entrance door position (if it exists)
    entrance_position = None
    for d in floor_plan.doors:
        if d.type_id == ROOM_TYPE_ENTRANCE:
            entrance_position = d.center
            break

    # Precompute target room max depth (diagonal)
    target_bounds = target_room.poly.bounds
    max_room_depth = math.hypot(
        target_bounds[2] - target_bounds[0],
        target_bounds[3] - target_bounds[1],
    )
    if max_room_depth < 1.0:
        max_room_depth = 1.0

    # Precompute target room perimeter for furniture score
    target_perimeter = target_room.poly.length
    if target_perimeter < 1.0:
        target_perimeter = 1.0

    best_score = -float('inf')
    best_center = None
    best_poly = None

    for i in range(num_steps + 1):
        if num_steps <= 1:
            dist = (usable_start + usable_end) / 2
        else:
            dist = usable_start + (usable_end - usable_start) * i / num_steps

        pt = segment.interpolate(dist)
        cx, cy = pt.x, pt.y

        # Build door polygon
        door_poly = build_door_polygon(
            cx=cx, cy=cy,
            ux=ux, uy=uy,
            nx=nx, ny=ny,
            half_width=half_door,
            half_depth=door_depth / 2,
        )

        if _collides_with_existing(door_poly, placed_doors, min_spacing):
            continue

        # ── Probe points: public side and private side ──────────────────
        pub_x = cx + in_to_current[0] * probe_offset
        pub_y = cy + in_to_current[1] * probe_offset
        priv_x = cx + in_to_target[0] * probe_offset
        priv_y = cy + in_to_target[1] * probe_offset

        # ── Compute isovist from the PUBLIC side of the door ────────────
        try:
            public_isovist = compute_isovist(
                (pub_x, pub_y), wall_segments, max_radius=isovist_radius
            )
        except Exception:
            continue

        # ────────────────────────────────────────────────────────────────
        # CRITERION 1: Privacy Score
        # How much of the private zone is exposed from the public side?
        # Lower exposure = higher privacy score.
        # ────────────────────────────────────────────────────────────────
        try:
            exposed = public_isovist.intersection(unified_private)
            exposed_area = exposed.area if not exposed.is_empty else 0.0
        except Exception:
            exposed_area = 0.0
        privacy_score = 1.0 - (exposed_area / private_area)

        # ────────────────────────────────────────────────────────────────
        # CRITERION 2: Bed Concealment Score
        # If current room is a bedroom, check if the bed is hidden.
        # ────────────────────────────────────────────────────────────────
        bed_score = 1.0  # default: perfect (not a bedroom or no bed)
        if current_room.type_id == ROOM_TYPE_BEDROOM:
            try:
                bed_rect = estimate_bed_position(
                    current_room.poly, (cx, cy),
                    bed_size_ratio=config.bed_size_ratio,
                )
                bed_visible = public_isovist.intersection(bed_rect)
                bed_visible_area = bed_visible.area if not bed_visible.is_empty else 0.0
                bed_area = bed_rect.area if bed_rect.area > 0 else 1.0
                bed_score = 1.0 - (bed_visible_area / bed_area)
            except Exception:
                bed_score = 0.5  # uncertain

        # ────────────────────────────────────────────────────────────────
        # CRITERION 3: Transition Zone Score
        # Measure sightline depth: how far can you see straight through
        # the door into the target room.  Shorter = better (natural turn).
        # ────────────────────────────────────────────────────────────────
        try:
            depth = measure_sightline_depth(
                (pub_x, pub_y), in_to_target, target_room.poly,
            )
        except Exception:
            depth = max_room_depth
        transition_score = 1.0 - min(1.0, depth / max_room_depth)

        # ────────────────────────────────────────────────────────────────
        # CRITERION 4: Furniture Placement Score
        # After placing a door, how much continuous wall is left in the
        # target room?  We approximate: longest remaining segment ratio.
        # ────────────────────────────────────────────────────────────────
        try:
            target_coords = list(target_room.poly.exterior.coords)
            max_wall_len = 0.0
            from shapely.geometry import LineString
            for j in range(len(target_coords) - 1):
                seg_j = LineString([target_coords[j], target_coords[j + 1]])
                # Subtract the door width if this segment overlaps the door
                if seg_j.distance(Point(cx, cy)) < door_width:
                    remaining = seg_j.length - door_width
                    max_wall_len = max(max_wall_len, remaining)
                else:
                    max_wall_len = max(max_wall_len, seg_j.length)
            furniture_score = max_wall_len / target_perimeter
        except Exception:
            furniture_score = 0.5

        # ────────────────────────────────────────────────────────────────
        # CRITERION 5: Public Exposure Penalty
        # If the entrance door can see through this door into the
        # private zone, penalise heavily.
        # ────────────────────────────────────────────────────────────────
        public_exposure_penalty = 0.0
        if entrance_position is not None:
            try:
                entrance_isovist = compute_isovist(
                    entrance_position, wall_segments,
                    max_radius=isovist_radius,
                )
                # Can the entrance see this door opening?
                if entrance_isovist.contains(Point(cx, cy)):
                    # The entrance can see the door.
                    # Now check if the private zone behind the door is also visible.
                    private_through_door = entrance_isovist.intersection(
                        unified_private
                    )
                    if not private_through_door.is_empty:
                        public_exposure_penalty = min(
                            1.0,
                            private_through_door.area / private_area,
                        )
            except Exception:
                public_exposure_penalty = 0.0

        # ── Weighted total ──────────────────────────────────────────────
        total = (
            config.privacy_weight          * privacy_score
            + config.bed_concealment_weight * bed_score
            + config.transition_weight      * transition_score
            + config.furniture_weight       * furniture_score
            - config.public_exposure_weight * public_exposure_penalty
        )

        if total > best_score:
            best_score = total
            best_center = (cx, cy)
            best_poly = door_poly

    if best_center is None:
        return (None, None, 0.0)

    return (best_center, best_poly, best_score)


# ── Adjacent room helpers ───────────────────────────────────────────────

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

    if rtype in (ROOM_TYPE_BEDROOM, ROOM_TYPE_KITCHEN, ROOM_TYPE_STORAGE):
        living_adj = [(r, g) for r, g in adjacents
                      if r.type_id == ROOM_TYPE_LIVING]
        if living_adj:
            return max(living_adj, key=lambda x: x[1].length)
        return max(adjacents, key=lambda x: x[1].length)

    if rtype == ROOM_TYPE_BATHROOM:
        living_adj = [(r, g) for r, g in adjacents
                      if r.type_id == ROOM_TYPE_LIVING]
        bed_adj = [(r, g) for r, g in adjacents
                   if r.type_id == ROOM_TYPE_BEDROOM]

        valid_living = [(r, g) for r, g in living_adj
                        if g.length > _door_width_hint()]
        if valid_living:
            return max(valid_living, key=lambda x: x[1].length)
        if bed_adj:
            return max(bed_adj, key=lambda x: x[1].length)
        return max(adjacents, key=lambda x: x[1].length)

    return max(adjacents, key=lambda x: x[1].length)


def _door_width_hint() -> float:
    """Quick estimate of door width for adjacency tests."""
    return 30.0


def _compute_door_position(
    segment, door_width: float, offset_ratio: float,
) -> Tuple[float, float]:
    """Compute door centre along a wall segment using offset ratio."""
    seg_len = segment.length
    half_door = door_width / 2
    min_dist = half_door
    max_dist = seg_len - half_door

    if max_dist <= min_dist:
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
