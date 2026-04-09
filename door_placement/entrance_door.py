"""
Main entrance door placement algorithm.

Finds the optimal point along the exterior boundary of the main living
room that maximises a normalised objective function balancing
visibility, privacy, and service-zone avoidance.

This module should be executed **after** internal doors have been placed
so the vision-cone penalties can account for what is actually visible
through internal doorways.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from door_placement.config import (
    EntranceDoorConfig,
    ROOM_TYPE_LIVING,
    ROOM_TYPE_BEDROOM,
    ROOM_TYPE_KITCHEN,
    ROOM_TYPE_BATHROOM,
    ROOM_TYPE_ENTRANCE,
    ROOM_TYPE_INT_DOOR,
)
from door_placement.models import Room, Door, FloorPlan
from door_placement.geometry_utils import (
    extract_segments,
    segment_tangent_normal,
    inward_normal,
    build_vision_cone,
    build_door_polygon,
)


# ── Public API ──────────────────────────────────────────────────────────

def place_entrance_door(
    floor_plan: FloorPlan,
    config: Optional[EntranceDoorConfig] = None,
) -> Optional[Door]:
    """Find the optimal main entrance location on the living room's
    exterior wall and place the door.

    Algorithm
    ---------
    1. Identify exterior walls of the main living room.
    2. Discretise each wall and evaluate every candidate point.
    3. At each point, project a vision cone inward and compute:
       • Reward  — visible living-room area
       • Penalty — proximity to private-wing centroid
       • Penalty — cone overlap with service rooms
       • Penalty — cone overlap with private rooms
       • Penalty — static proximity to service walls
       • Penalty — cone sees placed internal doors leading to bedrooms
    4. All thresholds are normalised to the floor plan's characteristic
       dimension, making the function resolution-independent.
    5. Select the point with the highest total score.

    Parameters
    ----------
    floor_plan : FloorPlan
    config : EntranceDoorConfig, optional

    Returns
    -------
    Door or None
    """
    if config is None:
        config = EntranceDoorConfig()

    main_living = floor_plan.main_living
    if main_living is None:
        print("[!] No living room found — cannot place entrance.")
        return None

    outer = floor_plan.outer_boundary
    if outer is None or outer.is_empty:
        print("[!] No outer boundary — cannot place entrance.")
        return None

    cd = floor_plan.characteristic_dimension

    # ── Precompute zone geometries ──────────────────────────────────────
    service_polys = [r.poly for r in floor_plan.rooms
                     if r.type_id in (ROOM_TYPE_KITCHEN, ROOM_TYPE_BATHROOM)]
    unified_service = unary_union(service_polys) if service_polys else Polygon()

    private_polys = [r.poly for r in floor_plan.rooms
                     if r.type_id == ROOM_TYPE_BEDROOM]
    unified_private = unary_union(private_polys) if private_polys else Polygon()

    priv_center = (unified_private.centroid
                   if not unified_private.is_empty
                   else main_living.centroid)

    # Internal doors that lead to private rooms
    private_door_polys = _get_private_door_polys(floor_plan)

    # ── Exterior walls of the living room ───────────────────────────────
    exterior_walls_geom = (
        main_living.poly.exterior
        .buffer(1)
        .intersection(outer.exterior)
        .simplify(5.0)
    )

    door_width = cd * config.width_ratio
    door_depth = cd * config.depth_ratio

    segments = extract_segments(exterior_walls_geom, min_length=door_width)
    if not segments:
        segments = extract_segments(exterior_walls_geom)
    if not segments:
        print("[!] No exterior wall segments found for the living room.")
        return None

    # ── Precompute ratio-based thresholds ───────────────────────────────
    cone_length     = cd * config.cone_length_ratio
    zone_radius     = cd * config.zone_penalty_radius_ratio
    service_radius  = cd * config.service_overlap_radius_ratio
    private_radius  = cd * config.private_overlap_radius_ratio
    static_radius   = cd * config.static_proximity_radius_ratio

    # ── For normalisation: compute max possible area reward ──────────────
    max_living_area = main_living.poly.area
    if max_living_area == 0:
        max_living_area = 1.0  # avoid division by zero

    # ── Evaluate every candidate position ───────────────────────────────
    best_door: Optional[_Candidate] = None

    for seg in segments:
        seg_len = seg.length
        slide_steps = max(1, int((seg_len - door_width) /
                                 config.slide_step_px))

        for step in range(slide_steps + 1):
            if slide_steps <= 1:
                dist = seg_len / 2
            else:
                dist = (door_width / 2) + step * (
                    (seg_len - door_width) / slide_steps
                )

            pt = seg.interpolate(dist)
            px, py = pt.x, pt.y

            # Inward normal
            nx, ny = inward_normal(seg, main_living.poly)

            # Vision cone
            cone = build_vision_cone(
                px, py, nx, ny,
                spread_deg=config.cone_spread_deg,
                length=cone_length,
            )

            # ── Reward: normalised visible living area ──────────────────
            visible_area = main_living.poly.intersection(cone).area
            area_score = (visible_area / max_living_area) * config.reward_area_weight

            # ── Penalty 1: proximity to private-wing centroid ───────────
            d_priv = pt.distance(priv_center)
            zone_pen = 0.0
            if d_priv < zone_radius:
                zone_pen = ((zone_radius - d_priv) / zone_radius
                            * config.zone_penalty_weight)

            # ── Penalty 2: cone overlaps service rooms ──────────────────
            service_pen = _overlap_penalty(
                cone, unified_service, pt,
                radius=service_radius,
                weight=config.service_overlap_weight,
                max_area=max_living_area,
            )

            # ── Penalty 3: cone overlaps private rooms ──────────────────
            private_pen = _overlap_penalty(
                cone, unified_private, pt,
                radius=private_radius,
                weight=config.private_overlap_weight,
                max_area=max_living_area,
            )

            # ── Penalty 4: static proximity to service walls ────────────
            static_pen = 0.0
            if not unified_service.is_empty:
                d_wall = pt.distance(unified_service)
                if d_wall < static_radius:
                    static_pen = ((static_radius - d_wall) / static_radius
                                  * config.static_proximity_weight)

            # ── Penalty 5: cone sees a private internal door ────────────
            int_door_pen = 0.0
            for dp in private_door_polys:
                if cone.intersects(dp):
                    int_door_pen += config.internal_door_privacy_weight
                    break  # one hit is enough

            # ── Total score ─────────────────────────────────────────────
            total = (area_score
                     - zone_pen
                     - service_pen
                     - private_pen
                     - static_pen
                     - int_door_pen)

            cand = _Candidate(
                point=(px, py),
                normal=(nx, ny),
                segment=seg,
                cone=cone,
                area_score=area_score,
                total=total,
            )

            if best_door is None or total > best_door.total:
                best_door = cand

    if best_door is None:
        print("[!] Could not find any valid entrance position.")
        return None

    # ── Build the door polygon ──────────────────────────────────────────
    px, py = best_door.point
    nx, ny = best_door.normal
    p1, p2 = best_door.segment.coords[0], best_door.segment.coords[-1]
    (ux, uy), _ = segment_tangent_normal(p1, p2)

    door_poly = build_door_polygon(
        px, py, ux, uy, nx, ny,
        half_width=door_width / 2,
        half_depth=door_depth / 2,
    )

    door = Door(
        type_id=ROOM_TYPE_ENTRANCE,
        poly=door_poly,
        center=best_door.point,
        normal=best_door.normal,
        score=best_door.total,
        connects=(main_living, None),
    )
    floor_plan.add_door(door)

    print(f"[*] Entrance placed — score {best_door.total:.2f} "
          f"(area_reward={best_door.area_score:.2f})")

    return door


# ── Private helpers ─────────────────────────────────────────────────────

class _Candidate:
    """Lightweight container for a candidate entrance position."""
    __slots__ = ("point", "normal", "segment", "cone",
                 "area_score", "total")

    def __init__(self, point, normal, segment, cone, area_score, total):
        self.point      = point
        self.normal     = normal
        self.segment    = segment
        self.cone       = cone
        self.area_score = area_score
        self.total      = total


def _overlap_penalty(
    cone: Polygon,
    zone: Polygon,
    origin: Point,
    radius: float,
    weight: float,
    max_area: float,
) -> float:
    """Distance-weighted, normalised overlap penalty."""
    if zone.is_empty:
        return 0.0

    overlap = cone.intersection(zone)
    if overlap.is_empty or overlap.area <= 0:
        return 0.0

    # Normalise overlap area by living-room area
    norm_area = overlap.area / max_area

    # Proximity weight: 1.0 right at the door, 0.1 at the edge of radius
    d = origin.distance(overlap)
    prox = max(0.1, (radius - d) / radius) if radius > 0 else 1.0

    return norm_area * prox * weight


def _get_private_door_polys(floor_plan: FloorPlan) -> List[Polygon]:
    """Return polygons of internal doors that connect to bedrooms."""
    result = []
    for door in floor_plan.doors:
        if door.type_id != ROOM_TYPE_INT_DOOR:
            continue
        a, b = door.connects
        if (a and a.type_id == ROOM_TYPE_BEDROOM) or \
           (b and b.type_id == ROOM_TYPE_BEDROOM):
            result.append(door.poly)
    return result
