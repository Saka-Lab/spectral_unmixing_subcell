import pandas as pd
from pathlib import Path
import re

def aggregate_and_normalize_similarity_to_self(
    data_root,
    similarity_subdir="subtracted_similarity",
    output_subdir="aggregated_similarity"
):
    data_root = Path(data_root)
    records = []

    pattern = re.compile(
        r"(FOV\d+)_(raw|group|full)_minus_GT_abs_similarity_to_self"
    )

    for fov_dir in data_root.iterdir():
        if not fov_dir.is_dir() or not fov_dir.name.startswith("FOV"):
            continue

        sim_dir = fov_dir / similarity_subdir
        if not sim_dir.exists():
            continue

        for csv_path in sim_dir.glob("*.csv"):
            match = pattern.search(csv_path.stem)
            if not match:
                continue

            fov, comparison = match.groups()

            df = pd.read_csv(csv_path, index_col=0)

            if "value" not in df.columns:
                raise KeyError(f"'value' column missing in {csv_path}")

            for row_label, value in df["value"].items():
                value = float(value)  # ensure numeric
                channel = row_label.split(" - ")[0]

                records.append({
                    "fov": fov,
                    "channel": channel,
                    "comparison": comparison,
                    "value": value,
                    "one_minus_value": 1.0 - value
                })

    if not records:
        raise RuntimeError("No similarity-to-self data found")

    df_all = pd.DataFrame(records)

    # -----------------------
    # Global normalization
    # -----------------------
    global_max = df_all["one_minus_value"].max()
    if global_max == 0:
        raise RuntimeError("Global max is zero, cannot normalize")

    df_all["value_norm"] = df_all["one_minus_value"] / global_max

    output_dir = data_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "all_fovs_similarity_to_self_normalized_long.csv"
    df_all.to_csv(long_path, index=False)

    return df_all