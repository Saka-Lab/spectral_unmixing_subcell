from pathlib import Path
import pandas as pd
import numpy as np
import re

def average_matrices_across_fovs(fov_dirs, folder_name, comparison_filter, filename_regex, channel_order=None):
    """
    For each comparison, average matrices across FOVs.

    Parameters:
    - fov_dirs: list of Path objects for FOV folders
    - folder_name: str, folder inside each FOV containing CSVs (e.g., "DAPI_normalized" or "subtracted")
    - comparison_filter: list of str, which comparisons to include (e.g., ["raw_vs_GT", "full_vs_GT"])
    - channel_order: list of str, optional, to enforce row/column order
    Returns:
    - dict: {comparison_name: average_df}
    """

    matrices_by_comparison = {comp: [] for comp in comparison_filter}

    # Match the comparison part in the filename
    pattern = re.compile(filename_regex)

    # Collect matrices
    for fov_dir in fov_dirs:
        folder_path = fov_dir / folder_name
        if not folder_path.is_dir():
            continue

        for csv_path in folder_path.glob("*.csv"):
            match = pattern.match(csv_path.name)
            if not match:
                continue

            comp = match.group(1)
            if comp not in comparison_filter:
                continue

            df = pd.read_csv(csv_path, index_col=0)

            # Enforce channel order safely (rows and columns separately)
            if channel_order is not None:
                row_labels = []
                col_labels = []

                for base in channel_order:
                    row_matches = [c for c in df.index if c.startswith(base)]
                    col_matches = [c for c in df.columns if c.startswith(base)]

                    row_labels.extend(row_matches)
                    col_labels.extend(col_matches)

                if not row_labels or not col_labels:
                    print(f"No matching channels in {csv_path.name}")
                    continue

                df = df.reindex(index=row_labels, columns=col_labels)

            matrices_by_comparison[comp].append(df)

    # Average matrices per comparison
    avg_matrices = {}
    for comp, dfs in matrices_by_comparison.items():
        if not dfs:
            print(f"No matrices found for comparison '{comp}'")
            continue

        avg_df = sum(dfs) / len(dfs)
        avg_matrices[comp] = avg_df

    return avg_matrices