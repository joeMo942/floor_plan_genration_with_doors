"""
Door Placement Engine for AI-Generated Floor Plans.

Pipeline order (architecturally correct):
    1. Load AI-generated room polygons
    2. Place INTERNAL doors   (most constrained — locked to shared walls)
    3. Place ENTRANCE door    (least constrained — any exterior living-room wall)
    4. Visualise & export
"""

from door_placement.pipeline import run_pipeline  # noqa: F401
