"""
Pipeline orchestrator — runs the full door-placement workflow.

    1. Load floor plan JSON
    2. Place internal doors  (most constrained)
    3. Place entrance door   (least constrained, uses internal-door info)
    4. Visualise & export
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from door_placement.config import PipelineConfig
from door_placement.floor_plan_loader import load_floorplan, save_floorplan
from door_placement.internal_doors import place_internal_doors
from door_placement.entrance_door import place_entrance_door
from door_placement.visualization import save_visualization


def run_pipeline(
    input_json: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    """Execute the complete door-placement pipeline.

    Parameters
    ----------
    input_json : str
        Path to the GSDiff-generated floor plan JSON.
    output_dir : str
        Directory for output files (JSON + images).
    config : PipelineConfig, optional
        Full configuration; uses sensible defaults if omitted.
    """
    if config is None:
        config = PipelineConfig()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DOOR PLACEMENT PIPELINE")
    print(f"{'='*60}")
    print(f"\n[1/4] Loading floor plan from {input_json}")
    floor_plan = load_floorplan(input_json, resolution=config.resolution)
    print(f"      Found {len(floor_plan.rooms)} rooms, "
          f"main living = {floor_plan.main_living}")

    # ── Step 2: Internal doors ──────────────────────────────────────────
    print(f"\n[2/4] Placing internal doors...")
    int_doors = place_internal_doors(floor_plan, config.internal_door)
    print(f"      Placed {len(int_doors)} internal door(s).")

    # ── Step 3: Save intermediate state ─────────────────────────────────
    intermediate_json = out / "floorplan_with_internal_doors.json"
    save_floorplan(floor_plan, intermediate_json)

    intermediate_img = out / "internal_doors_visualization.png"
    save_visualization(floor_plan, str(intermediate_img),
                       config.visualization)

    # ── Step 4: Entrance door ───────────────────────────────────────────
    print(f"\n[3/4] Placing entrance door...")
    entrance = place_entrance_door(floor_plan, config.entrance_door)
    if entrance:
        print(f"      Entrance score: {entrance.score:.2f}")

    # ── Step 5: Final export ────────────────────────────────────────────
    print(f"\n[4/4] Exporting final results...")
    final_json = out / "floorplan_with_all_doors.json"
    save_floorplan(floor_plan, final_json)

    final_img = out / "final_visualization.png"
    save_visualization(floor_plan, str(final_img), config.visualization)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output directory: {out}")
    print(f"{'='*60}\n")
