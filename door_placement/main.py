#!/usr/bin/env python3
"""
CLI entry point for the door-placement engine.

Usage
-----
    python -m door_placement.main --input path/to/floor.json --output ./results/

    # Or with custom resolution:
    python -m door_placement.main --input floor.json --output ./out/ --resolution 1024
"""

import argparse
import sys

from door_placement.config import PipelineConfig
from door_placement.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place doors on an AI-generated floor plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input custom_pred_5.json --output ./doors_output/
  %(prog)s -i floor.json -o ./out/ --resolution 1024
        """,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the GSDiff-generated floor plan JSON file.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Directory for output files (JSON + images).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution used during generation (default: 512).",
    )

    args = parser.parse_args()

    config = PipelineConfig(resolution=args.resolution)

    try:
        run_pipeline(
            input_json=args.input,
            output_dir=args.output,
            config=config,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
