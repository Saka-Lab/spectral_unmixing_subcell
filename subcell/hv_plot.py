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


def build_dashboard(umap_folder: str):
    cfg = load_config()

    if not os.path.exists(umap_folder):
        typer.echo(f"Error: Folder {umap_folder} does not exist.", err=True)
        raise typer.Exit(code=1)

    files = os.listdir(umap_folder)
    files = [f for f in files if f.endswith('.csv') or f.endswith('.tsv')]
    csv_files = [str(Path(umap_folder) / f) for f in files]

    if len(csv_files) == 0:
        typer.echo(f"Error: No CSV files found in folder {umap_folder}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nFound {len(csv_files)} files to plot:")
    tabs = pn.Tabs()

    for i, file in enumerate(csv_files):
        typer.echo(f" - {file}")
        layout, plot_umap = create_tab(file, cfg)
        tabs.append((file, layout))

    logger.info(f"Created {len(tabs)} tabs for the dashboard")
    return tabs, plot_umap


@app.command()
def serve(
        umap_folder: str = typer.Option(
            "data/subcell_results/10.0_99.99",
            "--umap-folder",
            "-u",
            help="Path to folder containing UMAP CSV/TSV files"
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
    dashboard, _ = build_dashboard(umap_folder)
    logger.info(f"Dashboard is now serving at port:{port}")
    pn.serve(dashboard, port=port, show=show, title="UMAP Dashboard")



@app.command()
def export(
        umap_folder: str = typer.Option(
            "data/subcell_results/10.0_99.99",
            "--umap-folder",
            "-u",
            help="Path to folder containing UMAP CSV/TSV files"
        ),
        output_file: str = typer.Option(
            "data/snapshot.svg",
            "--output-file",
            "-o",
            help="Output filename for the exported SVG (must include .svg at the end of the name)"
        )
):
    logger.info(f"Building dashboard from {umap_folder}...")
    _, plot_umap = build_dashboard(umap_folder)

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

    # Export (requires selenium + web driver)
    logger.info(f"Exporting to {output_file}...")
    export_svgs(bokeh_obj, filename=output_file)

    logger.info(f"✓ Successfully exported to {output_file}")


if __name__ == "__main__":
    app()