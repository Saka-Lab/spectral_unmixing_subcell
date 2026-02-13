from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import yaml
import json

from utils.plot_utils import load_data


app = typer.Typer(add_completion=False)

DEFAULT_CONDITION_ORDER = ["ActD", "Control", "SA"]
DEFAULT_MARKERS = ["NPM1", "G3BP1"]


def load_plot_configs():
    with open("configs/constants.yaml", "r", encoding="utf-8") as f:
        constants = yaml.safe_load(f)
    with open("configs/colormaps.json", "r", encoding="utf-8") as f:
        plotting_constants = json.load(f)
    rename_map = {
        f"soft_prob{i:02d}": subclass
        for i, subclass in enumerate(constants["subcell_classes"])
    }
    return constants, plotting_constants, rename_map


def save_plot(fig, filename: str, output_dir: Path, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi)
    plt.close(fig)


def condense_cell_probabilities(cell: dict, condensed_classes: dict) -> dict:
    condensed_cell = {}
    for marker, probs in cell.items():
        condensed_probs = {
            condensed_class: sum(probs.get(sc, 0) for sc in sub_classes)
            for condensed_class, sub_classes in condensed_classes.items()
        }
        condensed_cell[marker] = condensed_probs
    return condensed_cell


def prepare_box_dataframe(df: pd.DataFrame, condensed: bool, constants: dict) -> pd.DataFrame:
    drop_cols = [
        "cell_id",
        "image_name",
        "unique_cell_id",
        "experiment",
        "cell_cycle_phase",
        "WGA",
    ]
    box_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if not condensed:
        return box_df

    rows = []
    for _, row in box_df.iterrows():
        protein = row["protein"]
        condition = row["condition"]
        probs = {
            col: row[col]
            for col in row.index
            if col not in {"protein", "condition"}
        }
        condensed_values = condense_cell_probabilities(
            {protein: probs},
            constants["condensed_classes"],
        )[protein]
        rows.append((protein, condition, *condensed_values.values()))

    return pd.DataFrame(
        rows,
        columns=["protein", "condition", *constants["condensed_classes"].keys()],
    )


def plot_marker_distribution(
    box_df: pd.DataFrame,
    marker: str,
    condition_palette: dict,
    output_dir: Path,
) -> None:
    sub_df = box_df[box_df["protein"] == marker]
    if sub_df.empty:
        return

    sub_df = sub_df.drop(columns=["protein"])
    df_melt = sub_df.melt(
        var_name="Feature",
        value_name="Value",
        id_vars="condition",
    )

    present_conditions = df_melt["condition"].dropna().unique().tolist()
    hue_order = [cond for cond in DEFAULT_CONDITION_ORDER if cond in present_conditions]
    hue_order.extend(sorted(cond for cond in present_conditions if cond not in hue_order))
    if not hue_order:
        return

    fig, ax = plt.subplots(figsize=(20, 6))
    boxplot_width = 0.5
    features = df_melt["Feature"].drop_duplicates().tolist()

    sns.boxplot(
        x="Feature",
        y="Value",
        hue="condition",
        data=df_melt,
        dodge=True,
        width=boxplot_width,
        palette=condition_palette,
        ax=ax,
        hue_order=hue_order,
        fliersize=0,
        linewidth=1,
        boxprops={"alpha": 0.6},
    )

    n_conditions = len(hue_order)
    if n_conditions > 1:
        offset = boxplot_width / n_conditions
        offsets = np.linspace(-offset, offset, n_conditions)
    else:
        offsets = np.array([0.0])

    for i, cond in enumerate(hue_order):
        cond_data = df_melt[df_melt["condition"] == cond]
        medians = (
            cond_data.groupby("Feature", sort=False)["Value"].median().reindex(features)
        )
        x_vals = np.arange(len(features)) + offsets[i]
        ax.plot(
            x_vals,
            medians.values,
            color=condition_palette.get(cond, "#7f7f7f"),
            linestyle="--",
            linewidth=1.5,
            label="_nolegend_",
        )

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_xlim([-0.5, len(features) - 0.5])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Subcellular Classes")
    ax.set_title(f"Marker: {marker}")
    ax.legend(title="Conditions")

    plt.tight_layout()
    save_plot(fig, f"{marker}_boxplot_probabilities", output_dir=output_dir)


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/subcell_results/10.0_99.99"),
        "--input-dir",
        "-i",
        help="Directory containing SubCell result TSV files.",
    ),
    annotations_dir: Path = typer.Option(
        Path("annotations"),
        "--annotations-dir",
        "-a",
        help="Directory containing annotation files.",
    ),
    round_name: str = typer.Option(
        "round_5",
        "--round-name",
        "-r",
        help="Round name used to resolve input and annotation files.",
    ),
    interphase_only: bool = typer.Option(
        True,
        "--interphase-only/--all-phases",
        help="Restrict to interphase cells only.",
    ),
    condensed: bool = typer.Option(
        True,
        "--condensed/--full",
        help="Use condensed subcell classes instead of full class list.",
    ),
    model: str = typer.Option(
        "mae",
        "--model",
        "-m",
        help="Model suffix used in the results filename.",
    ),
) -> None:
    constants, plotting_constants, rename_map = load_plot_configs()

    df = load_data(
        input_dir=input_dir,
        annotations_dir=annotations_dir,
        round_name=round_name,
        model=model,
        interphase_only=interphase_only,
        rename_map=rename_map,
    )

    if "condition" in df.columns:
        df["condition"] = df["condition"].replace(
            {
                "ActinomycinD": "ActD",
                "SodiumArsenite": "SA",
                "Untreated": "Control",
            }
        )

    box_df = prepare_box_dataframe(df, condensed=condensed, constants=constants)

    output_dir = Path("plots") / round_name / "boxplots" / ("condensed" if condensed else "full")
    for marker in DEFAULT_MARKERS:
        plot_marker_distribution(
            box_df=box_df,
            marker=marker,
            condition_palette=plotting_constants["condition"],
            output_dir=output_dir,
        )


if __name__ == "__main__":
    app()
