from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MIN_WIDTH_UM = 0.12
MIN_SPACING_UM = 0.1
DEFAULT_DRC_LAYER = (1, 0)

from PIL import Image

try:  # pragma: no cover - optional dependency when running outside the agent stack
    from agent_r1.tool.base import BaseImageToolEnv, BaseTool
except ImportError:  # pragma: no cover - light-weight fallbacks for local testing
    class BaseTool:  # type: ignore[override]
        name = "base_tool"

        def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError("BaseTool.execute must be implemented by subclasses.")

    class BaseImageToolEnv:  # type: ignore[override]
        def __init__(self) -> None:
            pass

try:
    import gdsfactory as gf
    from gdsfactory.generic_tech import get_generic_pdk

    gf.config.rich_output()
    PDK = get_generic_pdk()
    PDK.activate()
    GDS_INSTALLED = True
    GDSComponent = gf.Component
except ImportError:  # pragma: no cover - defensive fallback
    gf = None
    GDS_INSTALLED = False
    GDSComponent = Any

try:  # pragma: no cover - optional dependency for geometry-accurate checks
    import klayout.db as kdb

    KLAYOUT_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive fallback when klayout isn't installed
    kdb = None
    KLAYOUT_AVAILABLE = False


def _iter_references(comp: GDSComponent) -> Iterable[Any]:  # pragma: no cover - helper
    if hasattr(comp, "insts"):
        for inst in comp.insts:
            yield inst
    elif hasattr(comp, "references"):
        for ref in comp.references:
            yield ref


def _ensure_reference_names(comp: GDSComponent) -> None:
    if not hasattr(comp, "named_instances") or comp.named_instances is None:
        comp.named_instances = {}

    for index, reference in enumerate(_iter_references(comp)):
        if not getattr(reference, "name", None):
            reference.name = f"p{index}"
        comp.named_instances[reference.name] = reference


def _component_polygons(component: GDSComponent, layer: Tuple[int, int]) -> List[Tuple[float, float, float, float]]:
    """Returns bounding boxes for every polygon on ``layer``.

    The helper inspects the raw polygon data provided by gdsfactory, converts it to
    axis-aligned bounding boxes, and filters out degenerate polygons.  The DRC
    routines only need bounding boxes because we restrict ourselves to simple width
    and spacing checks in this environment.
    """

    polygons: List[Tuple[float, float, float, float]] = []
    try:
        polygons_by_layer = component.get_polygons(by_spec=True)
    except Exception:  # pragma: no cover - defensive fallback
        polygons_by_layer = {}

    for spec, polys in polygons_by_layer.items():
        if tuple(spec) != layer:
            continue
        for poly in polys:
            if len(poly) < 3:
                continue
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            polygons.append(bbox)
    return polygons


def _bbox_distance(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute the minimum distance between two axis-aligned bounding boxes."""

    if b1[2] < b2[0]:
        dx = b2[0] - b1[2]
    elif b2[2] < b1[0]:
        dx = b1[0] - b2[2]
    else:
        dx = 0.0

    if b1[3] < b2[1]:
        dy = b2[1] - b1[3]
    elif b2[3] < b1[1]:
        dy = b1[1] - b2[3]
    else:
        dy = 0.0

    return (dx**2 + dy**2) ** 0.5


def _run_width_checks(
    bboxes: Sequence[Tuple[float, float, float, float]],
    min_width: float,
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for bbox in bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        min_feature = min(width, height)
        if min_feature < min_width:
            errors.append({"type": "min_width", "bbox": bbox, "measured": min_feature})
    return errors


def _run_spacing_checks(
    bboxes: Sequence[Tuple[float, float, float, float]],
    min_spacing: float,
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bbox_a = bboxes[i]
            bbox_b = bboxes[j]
            distance = _bbox_distance(bbox_a, bbox_b)
            if distance < min_spacing:
                errors.append(
                    {
                        "type": "min_spacing",
                        "bbox": (
                            min(bbox_a[0], bbox_b[0]),
                            min(bbox_a[1], bbox_b[1]),
                            max(bbox_a[2], bbox_b[2]),
                            max(bbox_a[3], bbox_b[3]),
                        ),
                        "measured": distance,
                        "pair": (i, j),
                    }
                )
    return errors


def _format_error_text(errors: Sequence[Dict[str, Any]]) -> str:
    if not errors:
        return "No DRC errors found."
    lines = []
    for error in errors:
        measured = error.get("measured")
        measured_txt = f" (measured {measured:.3f} um)" if isinstance(measured, float) else ""
        lines.append(f"ERROR: {error['type']} at {error['bbox']}{measured_txt}")
    return "\n".join(lines)


def _write_component_to_temp_gds(component: GDSComponent) -> Path:
    temp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    component.write_gds(str(temp_path))
    return temp_path


def _klayout_region_from_component(
    component: GDSComponent, layer: Tuple[int, int]
) -> Tuple[Any, Any]:
    if not KLAYOUT_AVAILABLE:
        raise RuntimeError("klayout is required for this operation.")

    temp_path = _write_component_to_temp_gds(component)
    try:
        layout = kdb.Layout()
        layout.read(str(temp_path))
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass

    top_cell = layout.top_cell()
    if top_cell is None:
        return layout, kdb.Region()

    layer_index = layout.layer(layer[0], layer[1])
    if layer_index < 0:
        return layout, kdb.Region()

    region = kdb.Region(top_cell.begin_shapes_rec(layer_index))
    return layout, region


def _edge_pairs_to_errors(edge_pairs: Any, error_type: str, dbu: float) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for pair in edge_pairs:
        bbox = pair.bbox()
        bbox_um = (
            float(bbox.left) * dbu,
            float(bbox.bottom) * dbu,
            float(bbox.right) * dbu,
            float(bbox.top) * dbu,
        )
        errors.append(
            {
                "type": error_type,
                "bbox": bbox_um,
                "measured": float(pair.distance()) * dbu,
            }
        )
    return errors


def _run_klayout_drc(
    component: GDSComponent,
    layer: Tuple[int, int],
    min_width: float,
    min_spacing: float,
) -> List[Dict[str, Any]]:
    layout, region = _klayout_region_from_component(component, layer)
    if region.is_empty():
        return []

    dbu = float(layout.dbu)
    errors: List[Dict[str, Any]] = []

    if min_width > 0:
        min_width_dbu = max(1, int(round(min_width / dbu)))
        width_pairs = region.width_check(min_width_dbu)
        errors.extend(_edge_pairs_to_errors(width_pairs, "min_width", dbu))

    if min_spacing > 0:
        min_spacing_dbu = max(1, int(round(min_spacing / dbu)))
        spacing_pairs = region.space_check(min_spacing_dbu)
        errors.extend(_edge_pairs_to_errors(spacing_pairs, "min_spacing", dbu))

    return errors


class DRCToolEnv(BaseImageToolEnv):
    """Environment that applies geometry tools to gdsfactory components."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        max_tool_response_length: int = 2048,
        data_root_dir: str | os.PathLike[str] = ".",
    ) -> None:
        super().__init__()
        self.tools: List[BaseTool] = list(tools)
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.max_tool_response_length = max_tool_response_length
        self.data_root_dir = Path(data_root_dir)

        self.batch_size: int = 0
        self.components: List[GDSComponent] = []
        self.op_counts: List[int] = []

    # ------------------------------------------------------------------
    # Environment lifecycle helpers
    # ------------------------------------------------------------------
    def reset(
        self,
        components: Sequence[GDSComponent] | None = None,
        gds_paths: Sequence[str] | None = None,
    ) -> None:
        if gf is None:
            raise RuntimeError("gdsfactory is required to load or manipulate GDS files.")

        loaded_components: List[GDSComponent] = []
        if gds_paths:
            for rel_path in gds_paths:
                abs_path = self.data_root_dir / rel_path
                if not abs_path.exists():
                    raise FileNotFoundError(f"GDS file not found at {abs_path}")
                component = gf.import_gds(str(abs_path))
                _ensure_reference_names(component)
                loaded_components.append(component)
        elif components:
            for component in components:
                _ensure_reference_names(component)
                loaded_components.append(component)
        else:
            raise ValueError("Must provide either 'components' or 'gds_paths'.")

        self.components = loaded_components
        self.batch_size = len(self.components)
        self.op_counts = [0] * self.batch_size

    # ------------------------------------------------------------------
    def get_drc_violations(self) -> List[Dict[str, Any]]:
        if gf is None:
            raise RuntimeError("gdsfactory is required to perform DRC checks.")

        batch_results: List[Dict[str, Any]] = []
        for component in self.components:
            if component is None:
                batch_results.append({
                    "count": 0,
                    "errors_text": "No component loaded.",
                    "errors_json": [],
                    "bboxes": [],
                    "component": component,
                })
                continue

            try:
                layer = DEFAULT_DRC_LAYER
                if KLAYOUT_AVAILABLE:
                    errors = _run_klayout_drc(component, layer, MIN_WIDTH_UM, MIN_SPACING_UM)
                else:
                    bboxes = _component_polygons(component, layer)
                    width_errors = _run_width_checks(bboxes, MIN_WIDTH_UM)
                    spacing_errors = _run_spacing_checks(bboxes, MIN_SPACING_UM)
                    errors = width_errors + spacing_errors
                errors_text = _format_error_text(errors)

                batch_results.append({
                    "count": len(errors),
                    "errors_text": errors_text,
                    "errors_json": errors,
                    "bboxes": [e["bbox"] for e in errors],
                    "component": component,
                })
            except Exception as exc:  # pragma: no cover - defensive
                batch_results.append({
                    "count": -1,
                    "errors_text": f"DRC check failed: {exc}",
                    "errors_json": [],
                    "bboxes": [],
                    "component": component,
                })
        return batch_results

    # ------------------------------------------------------------------
    def get_schematic(self, item_index: int) -> str:
        if item_index >= self.batch_size:
            raise IndexError(f"Index {item_index} is out of range for batch size {self.batch_size}.")
        component = self.components[item_index]
        if component is None:
            return "{}"
        try:
            return component.netlist()
        except AttributeError:
            return "{}"

    # ------------------------------------------------------------------
    def get_image(
        self,
        item_index: int,
        bbox: Tuple[float, float, float, float] | None = None,
    ) -> Image.Image:
        if item_index >= self.batch_size:
            raise IndexError(f"Index {item_index} is out of range for batch size {self.batch_size}.")

        component = self.components[item_index]
        if gf is None or component is None or not hasattr(gf, "plot"):
            return Image.new("RGB", (100, 100), color="white")

        kwargs: Dict[str, Tuple[float, float]] = {}
        if bbox:
            dx = (bbox[2] - bbox[0]) * 0.5
            dy = (bbox[3] - bbox[1]) * 0.5
            kwargs["xlim"] = (bbox[0] - dx, bbox[2] + dx)
            kwargs["ylim"] = (bbox[1] - dy, bbox[3] + dy)

        image_array = gf.plot.get_image(component, **kwargs)
        return Image.fromarray(image_array, "RGBA")

    # ------------------------------------------------------------------
    def step(
        self,
        raw_responses: Sequence[str] | str,
    ) -> Tuple[List[str], List[Image.Image], List[List[bool]], List[bool]]:
        if isinstance(raw_responses, str):
            raw_responses = [raw_responses]

        if self.batch_size == 0:
            raise RuntimeError("Environment has not been reset with any components.")

        if len(raw_responses) not in {1, self.batch_size}:
            raise ValueError(
                f"Expected {self.batch_size} responses (or 1 broadcast response) but received {len(raw_responses)}."
            )

        if len(raw_responses) == 1 and self.batch_size > 1:
            raw_responses = list(raw_responses) * self.batch_size

        batch_formatted_responses: List[str] = []
        batch_images: List[Image.Image] = []
        batch_successes: List[List[bool]] = []
        batch_active: List[bool] = []

        for index, raw_response in enumerate(raw_responses):
            component = self.components[index]
            tool_calls = self.extract_tool_calls(raw_response)

            if not tool_calls:
                batch_formatted_responses.append("")
                batch_images.append(self.get_image(index))
                batch_successes.append([])
                batch_active.append(False)
                continue

            tool_responses: List[str] = []
            tool_successes: List[bool] = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {}) or {}

                tool = self.tool_map.get(tool_name)
                if tool is None:
                    tool_responses.append(f"Error: Tool '{tool_name}' not found.")
                    tool_successes.append(False)
                    continue

                try:
                    result = tool.execute(args=tool_args, component=component)
                    response_text = result.get("content", json.dumps(result))
                    success = bool(result.get("success", True))
                except Exception as exc:  # pragma: no cover - runtime safety
                    response_text = f"Error executing tool '{tool_name}': {exc}"
                    success = False

                tool_responses.append(response_text)
                tool_successes.append(success)
                if success:
                    self.op_counts[index] += 1

            batch_formatted_responses.append(self.format_tool_response(tool_responses))
            batch_images.append(self.get_image(index))
            batch_successes.append(tool_successes)
            batch_active.append(True)

        return batch_formatted_responses, batch_images, batch_successes, batch_active

    # ------------------------------------------------------------------
    def stop(self, raw_responses: Sequence[str]) -> List[bool]:
        return ["<answer>" in response or not self.extract_tool_calls(response) for response in raw_responses]

    # ------------------------------------------------------------------
    def extract_tool_calls(self, raw_response: str) -> List[Dict[str, Any]]:
        try:
            match = re.search(r"<tool_code>(.*?)</tool_code>", raw_response, re.DOTALL)
            if not match:
                return []
            tool_calls_str = match.group(1).strip()
            tool_calls = json.loads(tool_calls_str)
            return tool_calls if isinstance(tool_calls, list) else [tool_calls]
        except (json.JSONDecodeError, AttributeError):
            return []

    # ------------------------------------------------------------------
    def format_tool_response(self, tool_responses: Sequence[str]) -> str:
        if not tool_responses:
            return ""
        response_data = json.dumps({"results": [{"content": resp} for resp in tool_responses]})
        if len(response_data) > self.max_tool_response_length:
            response_data = response_data[: self.max_tool_response_length] + "..."
        return f"<tool_response>{response_data}</tool_response>"


__all__ = ["DRCToolEnv", "GDS_INSTALLED", "KLAYOUT_AVAILABLE"]
