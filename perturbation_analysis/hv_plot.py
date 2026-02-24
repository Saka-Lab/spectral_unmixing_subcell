import os
from pathlib import Path

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


def load_config():
    """Load configuration files"""
    with open('configs/umap_config.yaml') as f:
        config_data = yaml.safe_load(f)

    with open('configs/colormaps.json') as f:
        colormaps = json.load(f)
        config_data['colormap_dict'] = colormaps

    return SimpleNamespace(**config_data)


def build_dashboard(input_folder: str, annotation_folder: str = None, output_tsv_dir: Path = None):
    cfg = load_config()

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

    for i, file in enumerate(csv_files):
        typer.echo(f" - {file}")
        layout, plot_umap = create_tab(file, cfg, annotation_folder, output_tsv_dir)
        tabs.append((file, layout))

    logger.info(f"Created {len(tabs)} tabs for the dashboard")
    return tabs, plot_umap


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
    dashboard, _ = build_dashboard(input_folder, annotation_folder, output_tsv_dir)
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
        )
):
    logger.info(f"Building dashboard from {input_folder}...")
    _, plot_umap = build_dashboard(input_folder, annotation_folder, output_tsv_dir)

    color_by_val = 'protein'
    alpha_val = 0.7
    filters_val = {
        "protein": ['G3BP1', 'alphaTUBULIN', 'NPM1'],
    }

    kwargs = {**filters_val}
    logger.info(f"Generating plot snapshot...")
    plot_snapshot = plot_umap(
        color_by=color_by_val,
        alpha=alpha_val,
        trigger=False,
        show_shape_legend=False,
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