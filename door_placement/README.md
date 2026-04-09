# Door Placement Engine

This project is a standalone, well-structured Python engine for dynamically placing internal and external doors within AI-generated floor plans (from GSDiff or similar tools).

## Architecture & Algorithm Differences

This engine executes door placement in the architecturally correct order:
1. **Internal Doors**: Placed first because they are strictly geometrically constrained to shared walls between adjacent rooms. They are intelligently spaced (offset from corners) and uniquely sized according to room type, avoiding constant midpoint placements.
2. **Main Entrance**: Placed second on the main living room's exterior. It uses an improved, **wall-aware** vision cone to assess what is actually visible. By placing the entrance *after* the internal doors, its privacy penalties can correctly determine if it "sees" directly into a bedroom through an open internal doorway.
3. **No hardcoded magic pixel thresholds**: Measurements and penalty thresholds are stored in `config.py` and are expressed as dynamic **ratios** relative to the floor plan's characteristic dimension (diagonal). This guarantees identical behaviour regardless of the image resolution (e.g. 256px or 1024px).

## Project Structure

```
├── __init__.py           # Package init
├── config.py             # Ratio-based configurations for internal/external doors and visuals
├── models.py             # Domain models (Room, WallSegment, Door, FloorPlan)
├── geometry_utils.py     # Reusable math routines (Segment extraction, vision cones, swing checks)
├── floor_plan_loader.py  # I/O functions for loading/saving GSDiff JSON format
├── internal_doors.py     # Smart topological search for internal door placements
├── entrance_door.py      # Vision-cone-based optimization for the main entrance door
├── visualization.py      # PIL rendering for the floor plans, zones, and placed doors
├── pipeline.py           # Orchestrator to run everything in the correct procedural order
├── main.py               # Command Line Interface (CLI) entry point
└── requirements.txt      # Project dependencies
```

## Installation

Ensure you are within the directory and install the necessary requirements to run the engine.

```bash
pip install -r requirements.txt
```

*(Note: Depending on your Python installation, if your system complains about system packages you may need to run `pip install --break-system-packages -r requirements.txt` or create a virtual environment first).*

## How to use

Run the pipeline from the command line using `main.py`. You simply provide the path to your AI-generated floor plan JSON and the destination folder for the modified JSON and preview images.

```bash
# Basic run:
python -m door_placement.main --input /path/to/custom_pred_X.json --output /path/to/output_dir/

# Setting a custom resolution dynamically:
python -m door_placement.main -i custom_pred.json -o ./results/ --resolution 1024
```

This will run the full sequential pipeline and write the completed floor plan (with placed doors coordinates appended) alongside visualization PNG files directly into the output directory.
