"""Script to build a DRC dataset with polygon name annotations."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, Iterable, List, Sequence, Tuple

import datasets
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

POLYGON_LABELS_KEY = "polygon_labels"
POLYGON_NAME_PROPERTY_ID = 1


def polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Return the centroid of a polygon described by ``points``.

    Parameters
    ----------
    points:
        Nx2 array-like containing the polygon vertices in order.
    """

    if len(points) == 0:
        raise ValueError("Cannot compute centroid of empty polygon.")

    # Ensure numpy array with float dtype for calculations
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    area_accumulator = 0.0
    cx_accumulator = 0.0
    cy_accumulator = 0.0

    for idx in range(len(pts)):
        jdx = (idx + 1) % len(pts)
        cross = x[idx] * y[jdx] - x[jdx] * y[idx]
        area_accumulator += cross
        cx_accumulator += (x[idx] + x[jdx]) * cross
        cy_accumulator += (y[idx] + y[jdx]) * cross

    area = area_accumulator / 2.0
    if abs(area) < 1e-9:
        # Fallback to arithmetic mean when polygon is degenerate
        centroid = pts.mean(axis=0)
        return float(centroid[0]), float(centroid[1])

    cx = cx_accumulator / (6.0 * area)
    cy = cy_accumulator / (6.0 * area)
    return float(cx), float(cy)


def add_named_polygon(
    component: gf.Component,
    points: Sequence[Sequence[float]],
    *,
    layer: Tuple[int, int] = (1, 0),
    name: str,
) -> None:
    """Add a polygon to ``component`` and remember its name for plotting.

    The polygon's centroid and layer index are stored in ``component.info`` so the
    plotting helper can recover the label later.
    """

    shape = component.add_polygon(points, layer=layer)
    points_array = np.asarray(points, dtype=float)
    centroid = polygon_centroid(points_array)

    # Remember the polygon label so we can annotate during plotting.
    labels: List[Dict[str, object]] = list(component.info.get(POLYGON_LABELS_KEY, []))
    layer_index = int(component.kcl.layout.layer(layer[0], layer[1]))
    labels.append(
        {
            "name": name,
            "layer_index": layer_index,
            "centroid": centroid,
        }
    )
    component.info[POLYGON_LABELS_KEY] = labels

    # Try to attach the name to the underlying KLayout shape for completeness.
    try:
        shape.set_property(POLYGON_NAME_PROPERTY_ID, name)
    except Exception:
        pass


def _build_label_lookup(component: gf.Component) -> List[Dict[str, object]]:
    labels = component.info.get(POLYGON_LABELS_KEY, [])
    if isinstance(labels, Iterable):
        return [dict(entry) for entry in labels]
    return []


def _find_polygon_label(
    label_entries: List[Dict[str, object]],
    layer_index: int,
    centroid: Tuple[float, float],
) -> str | None:
    for entry in label_entries:
        if entry.get("layer_index") != layer_index:
            continue
        stored_centroid = entry.get("centroid")
        if stored_centroid is None:
            continue
        stored_centroid = np.asarray(stored_centroid, dtype=float)
        if np.allclose(stored_centroid, np.asarray(centroid, dtype=float), atol=1e-6):
            return str(entry.get("name", "")) or None
    return None


def plot_with_labels_and_vertices(component_to_plot: gf.Component, title: str, file_path: str) -> None:
    """Plot ``component_to_plot`` annotating vertex coordinates and polygon names."""

    fig, ax = plt.subplots()

    all_polygons_by_layer = component_to_plot.get_polygons_points()
    label_entries = _build_label_lookup(component_to_plot)

    for layer_index, polygons in all_polygons_by_layer.items():
        color = "skyblue"
        edge_color = "black"
        if layer_index == 1:
            color = "#1f77b4"
        elif layer_index == 2:
            color = "#ff7f0e"

        for poly_points in polygons:
            points_array = np.asarray(poly_points, dtype=float)
            patch = Polygon(points_array, closed=True, color=color, alpha=0.5, ec=edge_color, lw=0.5)
            ax.add_patch(patch)

            for x_coord, y_coord in points_array:
                ax.text(
                    x_coord,
                    y_coord,
                    f"({x_coord:.1f}, {y_coord:.1f})",
                    ha="left",
                    va="bottom",
                    fontsize=5,
                    color="red",
                )

            centroid = polygon_centroid(points_array)
            label = _find_polygon_label(label_entries, int(layer_index), centroid)
            if label:
                ax.text(
                    centroid[0],
                    centroid[1],
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue",
                    weight="bold",
                )

    ax.set_title(title)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def create_drc_dataset(output_dir: str, num_samples: int, split: str) -> datasets.Dataset:
    """Creates a multi-modal dataset for the DRC task with annotated polygons."""

    if gf is None:  # pragma: no cover - defensive, mirrors original script.
        raise ImportError("gdsfactory is required to create the DRC dataset. Please install it.")

    data: List[Dict[str, object]] = []
    gds_dir = os.path.join(output_dir, "gds")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(gds_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for index in range(num_samples):
        component = gf.Component(f"{split}_clean_{index}")
        add_named_polygon(
            component,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            layer=(1, 0),
            name="p1",
        )

        gds_path_rel = os.path.join("gds", f"{split}_clean_{index}.gds")
        png_path_rel = os.path.join("png", f"{split}_clean_{index}.png")
        gds_path_abs = os.path.join(output_dir, gds_path_rel)
        png_path_abs = os.path.join(output_dir, png_path_rel)

        component.write_gds(gds_path_abs)
        plot_with_labels_and_vertices(component, "drc_clean_layout", png_path_abs)

        target_error_text = (
            "Create a DRC violation. Use op_split_polygon to split 'p1' "
            "and then use op_move_polygon to move one of the resulting halves "
            "to create a spacing violation (less than 0.1um)."
        )

        data.append(
            {
                "data_source": "drc",
                "ability": "drc_generation_and_fix",
                "split": split,
                "index": index,
                "initial_gds_path": gds_path_rel,
                "initial_image_path": png_path_rel,
                "target_error_text": target_error_text,
                "prompt": [{"role": "user", "content": "See image and task text."}],
                "reward_model": {"style": "rule", "ground_truth": ""},
            }
        )

    df = pd.DataFrame(data)
    return datasets.Dataset.from_pandas(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/drc_multimodal")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=10)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)

    if gf is None:
        print("Cannot run data preprocessing: gdsfactory is not installed.")
        return

    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Creating DRC training dataset at {local_dir}...")
    train_dataset = create_drc_dataset(local_dir, args.train_size, "train")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    print(f"Creating DRC validation dataset at {local_dir}...")
    val_dataset = create_drc_dataset(local_dir, args.val_size, "validation")
    val_dataset.to_parquet(os.path.join(local_dir, "validation.parquet"))

    print(f"\nDRC multi-modal datasets created in {local_dir}")
    print("A sample record:")
    print(train_dataset[0])


if __name__ == "__main__":
    main()
