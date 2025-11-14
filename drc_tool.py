from typing import Dict, Any, List
import sys  
sys.path.append('.')
from agent_r1.tool.base import BaseTool
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import gdsfactory as gf
from gdsfactory import get_layer
from gdsfactory.boolean import get_ref_shapes
from gdsfactory.typings import component
import klayout.db as kdb

from agent_r1.tool.base import BaseTool


_COLOR_CYCLE = plt.rcParams.get("axes.prop_cycle", None)


def _iter_references(comp: component) -> Iterable[gf.ComponentReference]:
    """Yield component references stored on the component."""

    if hasattr(comp, "insts"):
        for inst in comp.insts:
            yield inst


@dataclass(slots=True)
class _ReferenceMatch:
    reference: gf.ComponentReference
    index: int


def _find_reference_by_name(comp: component, name: str) -> _ReferenceMatch:
    for index, ref in enumerate(_iter_references(comp)):
        if ref.name == name:
            return _ReferenceMatch(reference=ref, index=index)
    raise ValueError(f"Component does not contain a reference named {name!r}.")


def plot_with_labels_and_vertices(component_to_plot: component, title: str):
    """Plot component polygons with labels and vertex coordinates."""

    polygons_by_layer = component_to_plot.get_polygons_points(by="tuple")
    fig, ax = plt.subplots()

    colors = (_COLOR_CYCLE.by_key()["color"] if _COLOR_CYCLE else ["tab:blue"])
    color_count = len(colors)
    color_index = 0

    for layer, polygons in polygons_by_layer.items():
        for poly in polygons:
            color = colors[color_index % color_count]
            color_index += 1

            patch = Polygon(poly, closed=True, facecolor="none", edgecolor=color)
            ax.add_patch(patch)

            centroid = poly.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                f"layer={layer}",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )

            for x, y in poly:
                ax.text(x, y, f"({x:.3f}, {y:.3f})", fontsize=8, color=color)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    return fig, ax


class DRCBaseTool(BaseTool):
    """Base class for tools that operate on gdsfactory components."""

    def execute(
        self, args: Dict[str, Any], component: component | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if component is None:
            raise ValueError("A component instance must be provided to execute the tool.")
        if not isinstance(args, dict):
            raise TypeError("Tool arguments must be provided as a dictionary.")

        result = self._execute(args, component)
        if result is None:
            result = {}
        return {"status": "success", **result}

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        raise NotImplementedError

    @staticmethod
    def _get_reference(component: component, name: str) -> gf.ComponentReference:
        return _find_reference_by_name(component, name).reference


class MovePolygonTool(DRCBaseTool):
    """Move a polygon (component reference) by a delta."""

    name = "op_move_polygon"
    description = "通过给定的增量 (dx, dy) 移动布局中的指定“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要移动的‘多边形’（实例）的名称。"},
            "dx": {"type": "number", "description": "x 方向的移动距离 (um)。"},
            "dy": {"type": "number", "description": "y 方向的移动距离 (um)。"},
        },
        "required": ["polygon_name", "dx", "dy"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        dx = float(args["dx"])
        dy = float(args["dy"])

        reference = self._get_reference(component, polygon_name)
        reference.dmove((dx, dy))
        return {"moved_reference": polygon_name, "dx": dx, "dy": dy}


class DeletePolygonTool(DRCBaseTool):
    """Delete a polygon (component reference) from the component."""

    name = "op_delete_polygon"
    description = "从布局中删除指定的“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要删除的‘多边形’（实例）的名称。"}
        },
        "required": ["polygon_name"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        reference_match = _find_reference_by_name(component, polygon_name)
        component.insts.remove(reference_match.reference)
        return {"deleted_reference": polygon_name}


class OffsetPolygonTool(DRCBaseTool):
    """Offset a polygon by a specific distance."""

    name = "op_offset_polygon"
    description = "通过给定距离偏移（收缩或增长）指定的“多边形”（实例）。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要偏移的‘多边形’（实例）的名称。"},
            "distance": {
                "type": "number",
                "description": "偏移距离 (um)。负数表示收缩，正数表示增长。",
            },
            "layer": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "GDS Layer [layer, purpose] 列表 (例如 [1, 0])。",
            },
        },
        "required": ["polygon_name", "distance", "layer"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        distance = float(args["distance"])
        layer_tuple = tuple(args["layer"])
        if len(layer_tuple) != 2:
            raise ValueError("Layer must be specified as [layer, purpose].")

        reference = self._get_reference(component, polygon_name)
        distance_dbu = component.kcl.to_dbu(distance)

        layer_index = get_layer(layer_tuple)
        region = get_ref_shapes(reference, layer_index)
        region = region.size(distance_dbu)

        component.insts.remove(reference)

        new_component = gf.Component(name=f"{polygon_name}_offset")
        new_component.kdb_cell.shapes(layer_index).insert(region)

        component.add_ref(new_component, name=polygon_name)
        return {"offset_reference": polygon_name, "distance": distance}


class SplitPolygonTool(DRCBaseTool):
    """Split a polygon into two pieces using a rectangular window."""

    name = "op_split_polygon"
    description = "使用由边界框定义的切割矩形来分割一个“多边形”（实例）。原始“多边形”被替换为两个新的结果“多边形”。"
    parameters = {
        "type": "object",
        "properties": {
            "polygon_name": {"type": "string", "description": "要分割的‘多边形’（实例）的名称。"},
            "split_line_bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "切割矩形的边界框 [xmin, ymin, xmax, ymax] (um)。",
            },
            "layer": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "GDS Layer [layer, purpose] 列表 (例如 [1, 0])。",
            },
        },
        "required": ["polygon_name", "split_line_bbox", "layer"],
    }

    def _execute(self, args: Dict[str, Any], component: component) -> Dict[str, Any] | None:
        polygon_name = args["polygon_name"]
        xmin, ymin, xmax, ymax = map(float, args["split_line_bbox"])
        layer_tuple = tuple(args["layer"])
        if len(layer_tuple) != 2:
            raise ValueError("Layer must be specified as [layer, purpose].")

        reference = self._get_reference(component, polygon_name)

        mask = gf.Component(name="split_mask")
        mask.add_polygon(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            layer=layer_tuple,
        )

        inside = gf.boolean(reference, mask, operation="and", layer=layer_tuple)
        outside = gf.boolean(reference, mask, operation="A-B", layer=layer_tuple)

        component.insts.remove(reference)

        new_refs: List[str] = []
        if inside.get_polygons():
            ref_inside = component.add_ref(inside, name=f"{polygon_name}_part1")
            new_refs.append(ref_inside.name)
        if outside.get_polygons():
            ref_outside = component.add_ref(outside, name=f"{polygon_name}_part2")
            new_refs.append(ref_outside.name)

        return {"split_reference": polygon_name, "new_references": new_refs}


def _create_demo_component() -> component:
    """Create a demo component with a single rectangle reference."""

    demo = gf.Component("drc_demo")
    rectangle = gf.components.rectangle(size=(40.0, 20.0), layer=(1, 0))
    demo.add_ref(rectangle, name="demo_rect")
    return demo


def _run_tool_and_plot(
    tool: DRCBaseTool,
    args: Dict[str, Any],
    comp: component,
    title: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute a tool, plot the result, and persist the figure."""

    result = tool.execute(args=args, component=comp)
    fig, _ax = plot_with_labels_and_vertices(comp, title=title)
    output_path = output_dir / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tool.name}] {title}: {result}")
    print(f"Saved plot to: {output_path.resolve()}")
    return result


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output_directory = Path("drc_demo_outputs")
    output_directory.mkdir(parents=True, exist_ok=True)

    demo_component = _create_demo_component()
    fig, _ = plot_with_labels_and_vertices(demo_component, "Initial component state")
    initial_path = output_directory / "initial_component_state.png"
    fig.savefig(initial_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {initial_path.resolve()}")

    move_tool = MovePolygonTool()
    _run_tool_and_plot(
        move_tool,
        {"polygon_name": "demo_rect", "dx": 10.0, "dy": 5.0},
        demo_component,
        "After move",
        output_directory,
    )

    offset_tool = OffsetPolygonTool()
    _run_tool_and_plot(
        offset_tool,
        {"polygon_name": "demo_rect", "distance": 2.0, "layer": [1, 0]},
        demo_component,
        "After offset",
        output_directory,
    )

    split_tool = SplitPolygonTool()
    split_result = _run_tool_and_plot(
        split_tool,
        {
            "polygon_name": "demo_rect",
            "split_line_bbox": [20.0, -5.0, 60.0, 25.0],
            "layer": [1, 0],
        },
        demo_component,
        "After split",
        output_directory,
    )

    new_refs = split_result.get("new_references", [])
    if new_refs:
        delete_tool = DeletePolygonTool()
        _run_tool_and_plot(
            delete_tool,
            {"polygon_name": new_refs[0]},
            demo_component,
            "After delete",
            output_directory,
        )

    print(
        "Demo complete. Generated plots demonstrate each tool operation on the sample"
        " component."
    )