"""
Centralised configuration for the door-placement engine.

Every threshold is expressed as a **ratio** of the floor-plan's
characteristic dimension (bounding-box diagonal) so the algorithm
works identically on 256 × 256 or 2048 × 2048 images.
"""

from dataclasses import dataclass, field
from typing import Dict
import math


# ── Room-type IDs (matching the GSDiff convention) ──────────────────────
ROOM_TYPE_LIVING     = 0
ROOM_TYPE_BEDROOM    = 1
ROOM_TYPE_STORAGE    = 2
ROOM_TYPE_KITCHEN    = 3
ROOM_TYPE_BATHROOM   = 4
ROOM_TYPE_BALCONY    = 5
ROOM_TYPE_EXTERNAL   = 6
ROOM_TYPE_STAIRS     = 99
ROOM_TYPE_ENTRANCE   = 100
ROOM_TYPE_INT_DOOR   = 101

# Human-readable names
ROOM_TYPE_NAMES: Dict[int, str] = {
    ROOM_TYPE_LIVING:   "Living",
    ROOM_TYPE_BEDROOM:  "Bedroom",
    ROOM_TYPE_STORAGE:  "Storage",
    ROOM_TYPE_KITCHEN:  "Kitchen",
    ROOM_TYPE_BATHROOM: "Bathroom",
    ROOM_TYPE_BALCONY:  "Balcony",
    ROOM_TYPE_EXTERNAL: "External",
    ROOM_TYPE_STAIRS:   "Internal Stairs",
    ROOM_TYPE_ENTRANCE: "Main Entrance",
    ROOM_TYPE_INT_DOOR: "Internal Door",
}


@dataclass
class InternalDoorConfig:
    """Parameters for internal (inter-room) doors."""

    # Door dimensions — ratios of the floor-plan characteristic dimension
    # (bounding-box diagonal).  Sensible defaults modelled on real-world
    # proportions for a typical apartment at ~512 px resolution.
    default_width_ratio:  float = 0.06    # ≈ 32 px on a 512-px plan
    default_depth_ratio:  float = 0.02    # ≈ 10 px frame thickness

    # Per-room-type width overrides (ratio of characteristic dim)
    width_ratios: Dict[int, float] = field(default_factory=lambda: {
        ROOM_TYPE_BEDROOM:  0.065,   # standard 80–90 cm
        ROOM_TYPE_BATHROOM: 0.052,   # narrower 60–70 cm
        ROOM_TYPE_STORAGE:  0.052,   # narrow
        ROOM_TYPE_KITCHEN:  0.065,   # standard kitchen door
    })

    # Placement offset: how far along the shared wall (from the nearer
    # corner) to place the door centre.  0.5 = midpoint (bad).
    # 0.30 leaves one long usable wall and one short pocket.
    offset_from_corner_ratio: float = 0.30

    # Buffer radius used to detect shared walls between rooms whose
    # AI-generated polygons may not perfectly touch.
    adjacency_buffer_px: float = 2.0

    # Minimum shared-area threshold (px²) for two rooms to be considered
    # adjacent after buffering.
    adjacency_area_threshold: float = 5.0

    # Minimum distance between any two placed doors (ratio of char dim).
    min_door_spacing_ratio: float = 0.04

    # ── Isovist-based privacy scoring ────────────────────────────────────
    # When True, the algorithm uses a proper isovist (visibility polygon)
    # computed via angular-sweep raycasting to score each candidate door
    # position on 5 privacy criteria.  This is the gold standard in
    # architectural spatial analysis.
    enable_isovist_scoring: bool = True

    # ── Multi-criteria scoring weights ──────────────────────────────────
    # W1: Privacy — minimise private-area exposure from public side
    privacy_weight: float = 4.0

    # W2: Bed concealment — bed should NOT be visible from entry gaze
    bed_concealment_weight: float = 3.0

    # W3: Transition zone — prefer positions that create a natural turn
    transition_weight: float = 2.0

    # W4: Furniture placement — maximise usable continuous wall length
    furniture_weight: float = 1.5

    # W5: Public exposure penalty — penalise entrance sightlines
    public_exposure_weight: float = 3.5

    # ── Isovist engine parameters ───────────────────────────────────────
    # Maximum isovist radius (ratio of characteristic dim)
    isovist_max_radius_ratio: float = 0.6

    # Estimated bed size (ratio of room's shortest dimension)
    bed_size_ratio: float = 0.25

    # How far from the wall centre to probe the public/private sides (px)
    probe_offset: float = 5.0

    # Slide step size (px) when sampling candidate positions
    isovist_slide_step_px: float = 5.0



@dataclass
class EntranceDoorConfig:
    """Parameters for the main entrance door."""

    # Door geometry
    width_ratio: float = 0.08    # ≈ 40 px — wider than internal doors
    depth_ratio: float = 0.03    # frame thickness

    # ── Vision-cone parameters ──────────────────────────────────────────
    cone_spread_deg:   float = 30.0    # half-angle of the cone
    cone_length_ratio: float = 0.40    # how far into the plan the cone reaches

    # ── Scoring weights (all dimensionless) ─────────────────────────────
    # REWARD:  visible living-room area in the cone
    reward_area_weight: float = 1.0

    # PENALTY: door too close to private-wing centroid
    zone_penalty_weight:       float = 2.0
    zone_penalty_radius_ratio: float = 0.35  # activate when closer than this

    # PENALTY: cone overlaps service rooms (kitchen / bathroom)
    service_overlap_weight:       float = 3.0
    service_overlap_radius_ratio: float = 0.42

    # PENALTY: cone overlaps private rooms (bedrooms)
    private_overlap_weight:       float = 2.5
    private_overlap_radius_ratio: float = 0.49

    # PENALTY: door physically too close to service-room walls
    static_proximity_weight:       float = 4.0
    static_proximity_radius_ratio: float = 0.11

    # PENALTY: entrance vision cone can see into a placed internal door
    # that leads to a private room
    internal_door_privacy_weight: float = 3.0

    # Number of test points per wall segment
    slide_step_px: float = 10.0


@dataclass
class VisualizationConfig:
    """Colours and line widths for the final image."""

    room_colors: Dict[int, tuple] = field(default_factory=lambda: {
        ROOM_TYPE_LIVING:   (244, 241, 222),
        ROOM_TYPE_BEDROOM:  (234, 182, 159),
        ROOM_TYPE_STORAGE:  (107, 112, 92),
        ROOM_TYPE_KITCHEN:  (224, 122, 95),
        ROOM_TYPE_BATHROOM: (95, 121, 123),
        ROOM_TYPE_BALCONY:  (242, 204, 143),
        ROOM_TYPE_ENTRANCE: (0, 200, 255),
        ROOM_TYPE_INT_DOOR: (50, 205, 50),
        ROOM_TYPE_STAIRS:   (255, 150, 50),
    })
    wall_color:    tuple = (150, 150, 150)
    outline_color: tuple = (0, 0, 0)
    outline_width: int   = 3
    cone_fill:     tuple = (200, 255, 200)
    cone_outline:  tuple = (0, 200, 0)
    service_overlap_fill: tuple = (255, 100, 100)
    private_overlap_fill: tuple = (180, 100, 255)


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    internal_door: InternalDoorConfig    = field(default_factory=InternalDoorConfig)
    entrance_door: EntranceDoorConfig    = field(default_factory=EntranceDoorConfig)
    visualization: VisualizationConfig   = field(default_factory=VisualizationConfig)
    resolution:    int                   = 512


def char_dim(outer_bounds: tuple) -> float:
    """Return the characteristic dimension (diagonal) of a bounding box.

    Parameters
    ----------
    outer_bounds : tuple
        (minx, miny, maxx, maxy) — e.g. from ``polygon.bounds``.
    """
    minx, miny, maxx, maxy = outer_bounds
    return math.hypot(maxx - minx, maxy - miny)
