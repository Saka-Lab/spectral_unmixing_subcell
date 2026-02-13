import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_avg_heatmaps(
    avg_matrices,
    output_dir,
    vmax=1.0
):
    """
    Plot heatmaps for averaged matrices.

    Parameters
    ----------
    avg_matrices : dict[str, pd.DataFrame]
        Dictionary: comparison_name -> average matrix.
    output_dir : str | Path
        Folder to save heatmaps.
    vmax : float
        Max value for heatmap color scaling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for comparison, df in avg_matrices.items():
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            df, annot=True, fmt=".2f", vmin=0, vmax=vmax, cmap="coolwarm",
            cbar=True, square=True
        )
        ax.set_ylabel("")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        out_path = output_dir / f"{comparison}_avg_heatmap.svg"
        plt.savefig(out_path, dpi=300, format='svg')
        plt.close()
        print(f"Saved {out_path}")
