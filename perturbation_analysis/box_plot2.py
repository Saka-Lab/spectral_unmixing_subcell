from bokeh.palettes import Category20
from utils.plot_utils import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
import typer
from typing import List, Optional
from loguru import logger

app = typer.Typer()


def save_plot(fig, filename, output_dir="plots", dpi=300):
    """Save a matplotlib figure and close it to free memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi)
    plt.close(fig)


def condense_cell_probabilities(cell, constants):
    """
    Condense subcellular classes into broader categories.
    """
    condensed_cell = {}
    for marker, probs in cell.items():
        condensed_probs = {
            condensed_class: sum(probs.get(sc, 0) for sc in sub_classes)
            for condensed_class, sub_classes in constants["condensed_classes"].items()
        }
        condensed_cell[marker] = condensed_probs
    return condensed_cell


def load_configs():
    """Load configuration files"""
    with open('configs/constants.yaml', 'r') as f:
        constants = yaml.safe_load(f)

    with open('configs/colormaps.json', 'r') as f:
        plotting_constants = json.load(f)

    return constants, plotting_constants


def setup_colormaps(constants):
    """Setup color mappings for markers"""
    original_palette = Category20[20]
    Cat20_1 = [original_palette[i] for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]
    marker_colormap = dict(zip(constants["markers"], Cat20_1[:len(constants["markers"])]))
    return marker_colormap


def create_rename_map(constants):
    """Create rename mapping for probability columns"""
    return {f'soft_prob{i:02d}': subclass for i, subclass in enumerate(constants["subcell_classes"])}


def process_dataframe(df, constants, condensed=True):
    """Process and prepare dataframe for plotting"""
    # Replace condition names
    if 'condition' in df.columns:
        df['condition'] = df['condition'].replace({'ActinomycinD': 'ActD', 'SodiumArsenite': 'SA'})

    # Drop unnecessary columns
    box_df = df.drop(columns=['cell_id', 'image_name', 'unique_cell_id'])
    if 'experiment' in box_df.columns:
        box_df = box_df.drop(columns=['experiment'])
    if 'cell_cycle_phase' in box_df.columns:
        box_df = box_df.drop(columns=['cell_cycle_phase'])
    if 'WGA' in box_df.columns:
        box_df = box_df.drop(columns=['WGA'])

    if condensed:
        logger.info("Condensing subcellular classes into broader categories")
        rows = []
        for index, row in box_df.iterrows():
            pr = row['protein']
            ac_type = row['condition']
            probs = {f: row[f] for f in row.index if f != 'protein' and f != 'condition'}
            cell_dict = {pr: probs}
            new_row = list(condense_cell_probabilities(cell_dict, constants)[pr].values())
            rows.append((pr, ac_type, *new_row))
        box_df = pd.DataFrame(rows, columns=['protein', 'condition', *constants["condensed_classes"].keys()])

    return box_df


def create_boxplot(box_df, marker, hue_order, plotting_constants, output_dir, condensed=True):
    """Create and save boxplot for a specific marker"""
    logger.info(f"Processing marker: {marker}")
    sub_df = box_df[box_df['protein'] == marker]
    sub_df = sub_df.drop(columns=['protein'])

    df_melt = sub_df.melt(var_name='Feature', value_name='Value', id_vars='condition')

    fig, ax = plt.subplots(figsize=(20, 6))
    boxplot_width = 0.5

    # Calculate number of conditions and offset per boxplot
    n_conditions = len(hue_order)
    offset = boxplot_width / n_conditions if n_conditions > 1 else 0

    # Map features to numeric positions
    features = df_melt['Feature'].unique()

    # Compute offsets for each condition
    offsets = np.linspace(-offset, offset, n_conditions)

    # Boxplot
    sns.boxplot(
        x='Feature',
        y='Value',
        hue='condition',
        data=df_melt,
        dodge=True,
        width=boxplot_width,
        palette=plotting_constants["condition"],
        ax=ax,
        hue_order=hue_order,
        fliersize=0,
        linewidth=1,
        boxprops={'alpha': 0.6}
    )

    # Add line plots with the means
    for i, cond in enumerate(hue_order):
        cond_data = df_melt[df_melt['condition'] == cond]
        means = cond_data.groupby('Feature')['Value'].median()
        y_vals = means[features].values
        x_vals = np.arange(len(means)) + offsets[i]

        ax.plot(
            x_vals,
            y_vals,
            label=f"{cond} median",
            color=list(plotting_constants["condition_colormap"].values())[i],
            linestyle='--'
        )

    # Fix x-axis ticks and labels
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_xlim([-0.5, len(features) - 0.5])

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.set_ylabel('Probability')
    ax.set_xlabel('Subcellular Classes')
    ax.legend(by_label.values(), by_label.keys(), title='Conditions')

    plt.title(f"Marker: {marker}")
    plt.tight_layout()

    # Save plot
    save_plot(fig, f"{marker}_boxplot_probabilities", output_dir=output_dir)
    logger.success(f"Saved boxplot for {marker} to {output_dir}")


@app.command()
def generate_boxplots(
        input_dir: Path = typer.Option(
            Path.cwd() / "perturbation_data" / "subcell_results"/ "10.0_99.99",
            "--input-dir",
            "-i",
            help="Directory containing subcellular classification results"
        ),
        annotations_dir: Path = typer.Option(
            Path("annotations"),
            "--annotations-dir",
            "-a",
            help="Directory containing annotation files"
        ),
        experiment_name: str = typer.Option(
            "20250610_umx-selected",
            "--round-name",
            "-r",
            help="Name of the experiment"
        ),
        markers: Optional[List[str]] = typer.Option(
            None,
            "--marker",
            "-m",
            help="Markers to plot (can be specified multiple times). Defaults to NPM1 and G3BP1"
        ),
        output_dir: Path = typer.Option(
            Path("perturbation_data/plots"),
            "--output-dir",
            "-o",
            help="Base output directory for plots"
        ),
        interphase_only: bool = typer.Option(
            True,
            "--interphase-only/--all-phases",
            help="Filter for interphase cells only"
        ),
        condensed: bool = typer.Option(
            True,
            "--condensed/--full",
            help="Use condensed subcellular classes"
        ),
        model: str = typer.Option(
            "mae",
            "--model",
            help="Model name/identifier"
        ),
        hue_order: Optional[List[str]] = typer.Option(
            None,
            "--hue-order",
            help="Order of conditions in plots (can be specified multiple times). Defaults to Untreated, ActD, SA"
        )
):
    logger.info("Starting boxplot generation")

    # Set defaults
    if markers is None:
        markers = ['NPM1', 'G3BP1']
    if hue_order is None:
        hue_order = ['Untreated', 'ActD', 'SA']

    # Load configurations
    logger.info("Loading configuration files")
    constants, plotting_constants = load_configs()
    marker_colormap = setup_colormaps(constants)
    RENAME_MAP = create_rename_map(constants)

    # Load data
    logger.info(f"Loading data from {input_dir}")
    df = load_data(input_dir, annotations_dir, experiment_name, model, interphase_only, RENAME_MAP)

    n_markers = df["protein"].nunique()
    n_cells = int(df.shape[0] / n_markers)
    logger.info(f"Round {experiment_name} contains results about {n_cells} cells")

    cell_phases = 'Interphase only' if interphase_only else 'All phases'
    logger.info(f"Cell phases: {cell_phases}")

    # Process dataframe
    box_df = process_dataframe(df, constants, condensed)

    # Setup output directory
    plot_output_dir = output_dir / experiment_name / 'boxplots' / ('condensed' if condensed else 'full')
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {plot_output_dir}")

    # Generate plots for each marker
    for marker in markers:
        if marker not in box_df['protein'].unique():
            logger.warning(f"Marker {marker} not found in data. Skipping.")
            continue

        create_boxplot(box_df, marker, hue_order, plotting_constants, plot_output_dir, condensed)

    logger.success(f"All boxplots generated successfully in {plot_output_dir}")


if __name__ == "__main__":
    app()