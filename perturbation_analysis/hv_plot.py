import os
from pathlib import Path
from enum import Enum

import typer
from utils.plot_utils import create_tab
import json
import panel as pn
import holoviews as hv
import yaml
from types import SimpleNamespace
from bokeh.io import export_svgs
from loguru import logger
from datetime import datetime


logger.add(f"logs/hv_plot/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
app = typer.Typer()

hv.extension('bokeh')
pn.extension()


DEFAULT_EXPORT_OPTIONS = {
    "color_by": "protein",
    "alpha": 0.7,
    "filters": {},
    "shape_by": "protein",
    "show_shape_legend": False,
    "highlighted_cells": [],
    "marker_types": {
        "protein": {},
    },
}

EXPORT_PRESETS = {
    "fig5c": {
        "color_by": "protein",
        "filters": {
            "CellCycle": ["Interphase"],
            "condition": ["Untreated"],
        },
        "shape_by": "protein",
        "show_shape_legend": False,
        "highlighted_cells": [],
        "marker_types": {
            "protein": {},
        }
    },
    "fig5d": {
        "color_by": "condition",
        "filters": {
            "CellCycle": ["Interphase"],
            "protein": ["G3BP1", "alphaTUBULIN", "NPM1"],
        },
        "shape_by": "protein",
        "show_shape_legend": True,
        "highlighted_cells": ["SA_FOV1_70", "ActD_FOV1_19", "Untreated_FOV1_102"],
        "marker_types": {
            "protein": {
                "alphaTUBULIN": "diamond",
                "NPM1": "square",
                "G3BP1": "circle",
            },
        },
    },
    "supp_fig5": {
        "color_by": "condition",
        "filters": {
            "CellCycle": ["Interphase"],
        },
        "shape_by": "protein",
        "show_shape_legend": False,
        "highlighted_cells": [],
        "marker_types": {
            "protein": {},
        },
    },
}


class ExportPreset(str, Enum):
    fig5c = "fig5c"
    fig5d = "fig5d"
    supp_fig5 = "supp_fig5"


def load_config(
        marker_type_overrides: dict | None = None,
        shape_by_override: str | None = None,
        highlighted_cells_override: list[str] | None = None,
):
    """Load configuration files"""
    with open('configs/umap_config.yaml') as f:
        config_data = yaml.safe_load(f)

    with open('configs/colormaps.json') as f:
        colormaps = json.load(f)
        config_data['colormap_dict'] = colormaps

    if marker_type_overrides:
        marker_types = config_data.setdefault("marker_types", {})
        marker_types.update(marker_type_overrides)

    if shape_by_override is not None:
        config_data["shape_by"] = shape_by_override

    if highlighted_cells_override is not None:
        config_data["highlighted_cells"] = highlighted_cells_override

    return SimpleNamespace(**config_data)


def build_dashboard(
        input_folder: str,
        annotation_folder: str = None,
        output_tsv_dir: Path = None,
        marker_type_overrides: dict | None = None,
        shape_by_override: str | None = None,
        highlighted_cells_override: list[str] | None = None,
):
    cfg = load_config(marker_type_overrides, shape_by_override, highlighted_cells_override)

    if not os.path.exists(input_folder):
        typer.echo(f"Error: Folder {input_folder} does not exist.", err=True)
        raise typer.Exit(code=1)

    files = os.listdir(input_folder)
    files = [f for f in files if f.endswith('.csv') or f.endswith('.tsv')]
    csv_files = [str(Path(input_folder) / f) for f in files]

    if len(csv_files) == 0:
        typer.echo(f"Error: No CSV files found in folder {input_folder}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nFound {len(csv_files)} files to plot:")
    tabs = pn.Tabs()
    filter_columns = []

    for file in csv_files:
        typer.echo(f" - {file}")
        layout, plot_umap, tab_filter_columns = create_tab(file, cfg, annotation_folder, output_tsv_dir)
        if not filter_columns:
            filter_columns = tab_filter_columns
        tabs.append((file, layout))

    logger.info(f"Created {len(tabs)} tabs for the dashboard")
    return tabs, plot_umap, filter_columns


def _build_export_options(preset: ExportPreset | None) -> dict:
    if preset is None:
        return {
            "color_by": DEFAULT_EXPORT_OPTIONS["color_by"],
            "alpha": DEFAULT_EXPORT_OPTIONS["alpha"],
            "filters": dict(DEFAULT_EXPORT_OPTIONS["filters"]),
            "shape_by": DEFAULT_EXPORT_OPTIONS["shape_by"],
            "show_shape_legend": DEFAULT_EXPORT_OPTIONS["show_shape_legend"],
            "highlighted_cells": list(DEFAULT_EXPORT_OPTIONS["highlighted_cells"]),
            "marker_types": DEFAULT_EXPORT_OPTIONS["marker_types"],
        }

    preset_options = EXPORT_PRESETS[preset.value]
    return {
        "color_by": preset_options["color_by"],
        "alpha": DEFAULT_EXPORT_OPTIONS["alpha"],
        "filters": dict(preset_options["filters"]),
        "shape_by": preset_options["shape_by"],
        "show_shape_legend": preset_options["show_shape_legend"],
        "highlighted_cells": preset_options["highlighted_cells"],
        "marker_types": dict(preset_options["marker_types"]),
    }


@app.command()
def serve(
        input_folder: str = typer.Option(
            "perturbation_data/subcell_results/10.0_99.99",
            "--umap-folder",
            "-u",
            help="Path to folder containing UMAP CSV/TSV files"
        ),
        annotation_folder: str = typer.Option(
            "perturbation_data/annotations",
            "--annotation-folder",
            "-a",
            help="Path to folder containing annotation CSV files (optional)",
        ),
        output_tsv_dir: Path = typer.Option(
            "perturbation_data/umap",
            "--output-tsv-dir",
            "-otd",
            help="Output directory for the exported tsv file with umap coords (must include .tsv at the end of the name)."
                 " The name will <experiment name>_<selected subcell model>.tsv."
        ),
        port: int = typer.Option(
            5006,
            "--port",
            "-p",
            help="Port to serve the dashboard on"
        ),
        show: bool = typer.Option(
            True,
            "--show/--no-show",
            help="Automatically open browser"
        )
):
    dashboard, _, _ = build_dashboard(input_folder, annotation_folder, output_tsv_dir)
    logger.info(f"Dashboard is now serving at port:{port}")
    pn.serve(dashboard, port=port, show=show, title="UMAP Dashboard")


@app.command()
def export(
        input_folder: str = typer.Option(
            "perturbation_data/subcell_results/10.0_99.99",
            "--input-folder",
            "-u",
            help="Path to folder containing subcell latent coords CSV/TSV files"
        ),
        annotation_folder: str = typer.Option(
            "perturbation_data/annotations",
            "--annotation-folder",
            "-a",
            help="Path to folder containing annotation CSV files (optional)",
        ),
        output_tsv_dir: Path = typer.Option(
            "perturbation_data/umap",
            "--output-tsv-dir",
            "-otd",
            help="Output directory for the exported tsv file with umap coords (must include .tsv at the end of the name)."
                 " The name will <experiment name>_<selected subcell model>.tsv."
        ),
        output_plot_file: Path = typer.Option(
            "perturbation_data/plots/umap/snapshot.svg",
            "--output-plot-file",
            "-o",
            help="Output filename for the exported SVG (must include .svg at the end of the name)"
        ),
        preset: ExportPreset | None = typer.Option(
            None,
            "--preset",
            "-p",
            help="Preset used for reproducing figures from the manuscript."
        )
):
    export_options = _build_export_options(preset)

    logger.info(f"Building dashboard from {input_folder}...")
    _, plot_umap, _ = build_dashboard(
        input_folder,
        annotation_folder,
        output_tsv_dir,
        marker_type_overrides=export_options["marker_types"],
        shape_by_override=export_options["shape_by"],
        highlighted_cells_override=export_options["highlighted_cells"],
    )

    color_by_val = export_options["color_by"]
    alpha_val = export_options["alpha"]
    filters_val = export_options["filters"]

    kwargs = {**filters_val}
    logger.info(f"Generating plot snapshot...")
    plot_snapshot = plot_umap(
        color_by=color_by_val,
        alpha=alpha_val,
        trigger=False,
        show_shape_legend=export_options["show_shape_legend"],
        **kwargs
    )

    logger.info(f"Converting to Bokeh format...")
    bokeh_obj = hv.render(plot_snapshot)
    bokeh_obj.output_backend = "svg"

    output_plot_file.parent.mkdir(parents=True, exist_ok=True)
    # Export (requires selenium + web driver)
    logger.info(f"Exporting to {output_plot_file}...")
    export_svgs(bokeh_obj, filename=str(output_plot_file))

    logger.info(f"✓ Successfully exported to {output_plot_file}")


if __name__ == "__main__":
    app()
