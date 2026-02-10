from bokeh.palettes import Category20
from utils.utils import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json


def save_plot(fig, filename, output_dir = "plots", dpi= 300):
    """Save a matplotlib figure and close it to free memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi)
    plt.close(fig)

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

with open('configs/constants.yaml', 'r') as f:
    constants = yaml.safe_load(f)

with open('configs/colormaps.json', 'r') as f:
    plotting_constants = json.load(f)


original_palette = Category20[20]
Cat20_1 = [original_palette[i] for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]

marker_colormap = dict(zip(constants["markers"], Cat20_1[:len(constants["markers"])]))


RENAME_MAP = {f'soft_prob{i:02d}': subclass for i, subclass in enumerate(constants["subcell_classes"])}


input_dir = Path('subcell_results/10.00_99.99')
annotations_dir = Path('annotations')
round_name = 'round_5_3x5'

interphase_only = True
condensed = True
model = 'mae'

cell_phases = 'Interphase only' if interphase_only else 'All phases'


df = load_data(input_dir, annotations_dir, round_name, model, interphase_only, RENAME_MAP)
n_markers = df["protein"].nunique()
print(f"Round {round_name} contains results about {int(df.shape[0]/n_markers)} cells.")

if 'condition' in df.columns:
    df['condition'] = df['condition'].replace({'ActinomycinD': 'ActD', 'SodiumArsenite': 'SA'})



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
        new_row = list(condense_cell_probabilities(cell_dict)[pr].values())
        rows.append((pr, ac_type, *new_row))
    box_df = pd.DataFrame(rows, columns=['protein', 'condition', *constants["condensed_classes"].keys()])


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
    save_plot(fig, f"{marker}_boxplot_probabilities", output_dir=output_dir)
