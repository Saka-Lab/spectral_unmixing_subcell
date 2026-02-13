import pandas as pd
from pathlib import Path

def aggregate_and_normalize_row_sums(
    data_root: Path,
    row_sums_subdir="row_sums",
    output_subdir="aggregated_normalized"
):
    """
    Aggregate row sums across all FOVs and normalize.
    
    Normalization:
    1. Divide all values by 14 (number of channels)
    2. Divide by global max across all FOVs, comparisons, and channels
    """

    data_root = Path(data_root)
    aggregated_rows = []

    # Loop over all FOV folders
    fov_dirs = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("FOV"))

    for fov_dir in fov_dirs:
        row_sums_dir = fov_dir / row_sums_subdir
        if not row_sums_dir.exists():
            continue

        # Process each comparison CSV
        for csv_path in row_sums_dir.glob("*.csv"):
            # Determine comparison type from filename
            fname = csv_path.stem  # e.g. FOV1_raw_vs_GT_minus_GT_row_sum
            comparison = None
            for ref in ["raw", "group", "full"]:
                if f"_{ref}_" in fname:
                    comparison = ref
                    break
            if comparison is None:
                continue

            df = pd.read_csv(csv_path, index_col=0)
            # Convert to long format
            for channel_label, value in df["row_sum"].items():
                aggregated_rows.append({
                    "FOV": fov_dir.name,
                    "channel": channel_label,
                    "comparison": comparison,
                    "value": float(value)
                })

    if not aggregated_rows:
        raise ValueError("No row sums found in any FOVs.")

    # Build DataFrame
    master_df = pd.DataFrame(aggregated_rows)

    # Step 1: divide by number of channels (14)
    master_df["value_div14"] = master_df["value"] / 14.0

    # Step 2: normalize by global max
    global_max = master_df["value_div14"].max()
    master_df["value_norm"] = master_df["value_div14"] / global_max

    # Output directory
    output_dir = data_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save master table
    output_path = output_dir / "all_fovs_row_sums_normalized_long.csv"
    master_df.to_csv(output_path, index=False)

    print(f"Aggregated and normalized table saved to: {output_path}")

    # Optional: also return pivoted table for convenience
    pivot_df = master_df.pivot_table(
        index=["FOV", "channel"],
        columns="comparison",
        values="value_norm"
    ).reset_index()
    pivot_path = output_dir / "all_fovs_row_sums_normalized_wide.csv"
    pivot_df.to_csv(pivot_path, index=False)
    print(f"Wide-format table saved to: {pivot_path}")

    return master_df, pivot_df
