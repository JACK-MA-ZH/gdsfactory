"""End-to-end smoke test for the DRC data and environment pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from drc import DRCToolEnv
from drc_data_preprocess import create_drc_dataset
from drc_tool import MovePolygonTool, SplitPolygonTool


def _format_tool_calls(calls: list[dict]) -> str:
    return f"<tool_code>{json.dumps(calls)}</tool_code>"


def main() -> None:
    data_root = Path("demo_drc_data")
    dataset = create_drc_dataset(str(data_root), num_samples=1, split="demo")
    record = dataset[0]
    gds_rel_path = record["initial_gds_path"]

    env = DRCToolEnv(
        tools=[SplitPolygonTool(), MovePolygonTool()],
        max_tool_response_length=2048,
        data_root_dir=str(data_root),
    )
    env.reset(gds_paths=[gds_rel_path])

    split_calls = [
        {
            "name": "op_split_polygon",
            "arguments": {
                "polygon_name": "p1",
                "split_line_bbox": [0.5, -0.5, 0.6, 1.5],
                "layer": [1, 0],
            },
        }
    ]
    split_response = env.step(_format_tool_calls(split_calls))
    print("Split response:", split_response[0][0])

    move_calls = [
        {
            "name": "op_move_polygon",
            "arguments": {"polygon_name": "p1_part1", "dx": 0.2, "dy": 0.0},
        }
    ]
    move_response = env.step(_format_tool_calls(move_calls))
    print("Move response:", move_response[0][0])

    print("DRC violations after operations:")
    print(env.get_drc_violations()[0]["errors_text"])


if __name__ == "__main__":
    main()
