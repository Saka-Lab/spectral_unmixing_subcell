from itertools import cycle
from bokeh.palettes import Category20
from utils.utils import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

with open('configs/constants.yaml', 'r') as f:
    constants = yaml.safe_load(f)

with open('configs/plot_config.yaml', 'r') as f:
    plotting_constants = yaml.safe_load(f)

linestyles = cycle(['-', '--', '-.', ':'])
original_palette = Category20[20]
Cat20_1 = [original_palette[i] for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]

marker_colormap = dict(zip(constants["markers"], Cat20_1[:len(constants["markers"])]))
PROBABILITY_PREFIX = constants["probability_prefix"]


RENAME_MAP = {
    f'soft_prob{i:02d}': subclass
    for i, subclass in enumerate(constants["subcell_classes"])
}


class CellProbabilityPlotter:
    """
    A utility for computing and plotting average subcellular localization probabilities.
    """

    def __init__(
        self,
        df,
        marker_colormap,
        plot_markers=plotting_constants["plot_markers"],
        plot_linestyles=plotting_constants["plot_linestyles"],
        plot_facecolors=plotting_constants["plot_facecolors"],
        plot_edgecolors=plotting_constants["plot_edge_colors"],
        marker_size: int = plotting_constants["marker_size"],
        linewidth: float = plotting_constants["linewidth"],
        default_prob_prefix: str = "soft_prob",
    ) -> None:
        self.df = df
        self.marker_colormap = marker_colormap
        self.plot_markers = plot_markers
        self.plot_linestyles = plot_linestyles
        self.plot_facecolors = plot_facecolors
        self.plot_edgecolors = plot_edgecolors
        self.marker_size = marker_size
        self.linewidth = linewidth
        self.default_prob_prefix = default_prob_prefix
        self.subcell_classes = constants["subcell_classes"]

    def __repr__(self):
        return f"CellProbabilityPlotter(df={self.df.shape})"

    def __str__(self):
        return f"CellProbabilityPlotter with {self.df.shape[0]} rows and {self.df.shape[1]} columns"

    def compute_average_cell(self, df_subset: "pd.DataFrame", markers, condensed: bool = False):
        """
        Compute average subcellular probabilities for the given markers.
        Optionally return condensed probabilities.
        """
        prob_cols = list(RENAME_MAP.values())
        avg_cell = {}

        for marker in markers:
            marker_df = df_subset[df_subset["protein"] == marker]
            if marker_df.empty:
                continue
            avg_probs = marker_df[prob_cols].mean().to_numpy()
            avg_cell[marker] = dict(zip(self.subcell_classes, avg_probs))

        return (self.condense_cell_probabilities(avg_cell) if condensed else avg_cell)

    def plot_cell_probabilities(
        self, cell, title=None, marker='all', condition=None,
        legend: bool = True,
        y_max: float = 0.5,
        linestyle= None, ax=None, marker_style=None,
        plot_condensed: bool = False,
    ):
        """
        Plot probabilities for one marker.
        If plot_condensed=True, the cell dict will be condensed before plotting.
        """
        if plot_condensed:
            cell = self.condense_cell_probabilities(cell)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        pt_linestyle = linestyle or self.plot_linestyles.get(condition, "-")

        x_labels = list(cell.keys())
        x = np.arange(len(x_labels))

        probabilities = list(cell.values())

        ax.plot(
            x + 0.5,
            probabilities,
            label=f"{marker} - {condition}",
            color=self.marker_colormap.get(marker, "gray"),
            marker=marker_style or self.plot_markers.get(condition, "o"),
            linestyle=pt_linestyle,
            markersize=self.marker_size,
            markerfacecolor=self.plot_facecolors.get(condition, "white"),
            markeredgecolor=self.plot_edgecolors.get(condition, "black"),
            linewidth=self.linewidth,
        )

        ax.set(
            xlabel="Subcellular Classes",
            ylabel="Probabilities",
            xlim=[-0.5, len(x_labels) + 1],
            ylim=[0, 1.1 * y_max],
        )
        if title:
            ax.set_title(title)
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
        if legend:
            ax.legend(loc="best")

        plt.tight_layout()
        return ax

    @staticmethod
    def save_plot(fig, filename, output_dir = "plots", dpi= 300) -> None:
        """Save a matplotlib figure and close it to free memory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{filename}.png", dpi=dpi)
        plt.close(fig)

    @staticmethod
    def condense_cell_probabilities(cell):
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

    @staticmethod
    def avg_cell_by_condition(plotter, df, marker: str, condensed: bool):
        avg_cells = {}
        #for condition in df['condition'].unique():
        for condition in ['Untreated', 'ActD', 'SA']:
            df_prot = df[(df['condition'] == condition) & (df['protein'] == marker)]
            if df_prot.empty:
                print(f"No data for marker {marker} in condition {condition}")
                # create empty cell
                avg_cells[condition] = {marker: {k: 0.0 for k in plotter.subcell_classes}}
            else:
                avg_cell = plotter.compute_average_cell(df_prot, [marker])
                avg_cells[condition] = avg_cell
            if condensed:
                avg_cells[condition] = CellProbabilityPlotter.condense_cell_probabilities(avg_cells[condition])
        return avg_cells


condensed = True

input_dir = Path('subcell_results/10.00_99.99')
annotations_dir = Path('annotations')
model = 'mae'

interphase_only = True
cell_phases = 'Interphase only' if interphase_only else 'All phases'


round_name = 'round_5_3x5'

df = load_data(input_dir, annotations_dir, round_name, model, interphase_only, RENAME_MAP)
n_markers = df["protein"].nunique()
print(f"Round {round_name} contains results about {int(df.shape[0]/n_markers)} cells.")


if 'condition' in df.columns:
    df['condition'] = df['condition'].replace({
        'ActinomycinD': 'ActD',
        'SodiumArsenite': 'SA'
    })
plotter = CellProbabilityPlotter(df)


## If there are multiple conditions they will be plotted together

print('Plotting average cells...')
conditions = df['condition'].unique()
for marker in df['protein'].unique():
    fig, ax = plt.subplots(figsize=(10, 4))
    y_max = 0
    #for condition in conditions:
    for condition in ['Untreated', 'ActD', 'SA']:
        avg_cell = CellProbabilityPlotter.avg_cell_by_condition(plotter, df, marker, condensed)[condition][marker]
        # Find y_max across all conditions
        y_max = max(*list(avg_cell.values()), y_max)

        # Plot
        plotter.plot_cell_probabilities(
            avg_cell,
            marker=marker,
            condition=condition,
            ax=ax,
            y_max=y_max,
            title=f"{round_name} -- {marker} - {cell_phases}",
            legend=True
        )

    # Save plot
    output_dir = Path('plots') / round_name / 'average_cells' / ('condensed' if condensed else 'full')
    # check if output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    CellProbabilityPlotter.save_plot(fig, f"{marker}_average_probabilities", output_dir=output_dir)


print('Plotting single cells...')
max_cells = 1
# there is no 1 to 1 correspondence between cells for each condition so each condition/cell will be plotted separately
conditions = df['condition'].unique()
for marker in constants["markers"]:
    for condition in conditions:
        df_condition = df[(df['condition'] == condition) & (df['protein'] == marker)]
        c = 1
        for idx, row in df_condition.iterrows():
            if c > max_cells:
                break
            fig, ax = plt.subplots(figsize=(10, 4))
            cell_id = row['unique_cell_id']
            if 'cell_cycle_phase' in row:
                phase = row['cell_cycle_phase']
            else:
                phase = 'Unknown cell phase'
            cell = {
                marker: dict(zip(
                    constants["subcell_classes"],
                    row[RENAME_MAP.values()]
                ))
            }

            if condensed:
                cell = plotter.condense_cell_probabilities(cell)
                
            y_max = max(cell[marker].values())
            plotter.plot_cell_probabilities(
                cell[marker],
                condition=condition,
                marker=marker,
                ax=ax,
                y_max=y_max,
                title=f"{round_name} -- {cell_id} - {marker} - {phase}",
                legend=True
            )
            c += 1
            # Save plot
            output_dir = Path('plots') / round_name / 'single_cells' / ('condensed' if condensed else 'full') / marker / condition
            # check if output_dir exists
            output_dir.mkdir(parents=True, exist_ok=True)
            CellProbabilityPlotter.save_plot(fig, f"{cell_id}_probabilities", output_dir=output_dir)


print('Plotting boxplots')

box_df = df.drop(columns=['cell_id', 'image_name', 'unique_cell_id'])
if 'experiment' in box_df.columns:
    box_df = box_df.drop(columns=['experiment'])
if 'cell_cycle_phase' in box_df.columns:
    box_df = box_df.drop(columns=['cell_cycle_phase'])
if 'WGA' in box_df.columns:
    box_df = box_df.drop(columns=['WGA'])

if condensed:
    rows = []
    for index, row in box_df.iterrows():
        pr = row['protein']
        ac_type = row['condition']
        probs = {f: row[f] for f in row.index if f != 'protein'}
        cell_dict = {pr: probs}
        new_row = list(CellProbabilityPlotter.condense_cell_probabilities(cell_dict)[pr].values())
        rows.append((pr, ac_type, *new_row))
    box_df = pd.DataFrame(rows, columns=['protein', 'condition', *constants["condensed_classes"].keys()])


hue_order = box_df['condition'].unique()
hue_order = ['Untreated', 'ActD', 'SA']
boxplot_markers = ['NPM1', 'G3BP1']

for marker in boxplot_markers:
    print(f"Processing marker: {marker}")
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


    # Boxplot instead
    sns.boxplot(
        x='Feature',
        y='Value',
        hue='condition',
        data=df_melt,
        dodge=True,
        width=boxplot_width,
        palette=plotting_constants["condition_colormap"],
        ax=ax,
        hue_order=hue_order,
        fliersize=0,
        linewidth=1,
        boxprops={'alpha':0.6}
    )

    # add line plots with the means
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
    ax.set_xlim([-0.5, len(features)-0.5])

    # Create custom legend for points and boxplots
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate legend entries (points and boxplot hues share colors)
    by_label = dict(zip(labels, handles))
    ax.set_ylabel('Probability')
    ax.set_xlabel('Subcellular Classes')
    ax.legend(by_label.values(), by_label.keys(), title='Conditions')


    plt.title(f"Marker: {marker}")
    plt.tight_layout()
    # Save plot
    output_dir = Path('plots') / round_name / 'boxplots' / ('condensed' if condensed else 'full')
    # check if output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    CellProbabilityPlotter.save_plot(fig, f"{marker}_boxplot_probabilities", output_dir=output_dir)
