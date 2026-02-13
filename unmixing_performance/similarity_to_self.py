import pandas as pd
from pathlib import Path

def _base_channel(label: str) -> str:
    """
    Extract base channel name from labels like:
    'DAPI - Gr1', 'DAPI - raw', 'ALEXA 594 - full'
    """
    return label.split(" - ")[0].strip()


def similarity_to_self_all_matrices(
    subtracted_dir,
    output_subdir="subtracted_similarity"
):
    """
    For each subtracted matrix, extract similarity-to-self values
    by matching base channel names between rows and columns.
    """

    subtracted_dir = Path(subtracted_dir)
    output_dir = subtracted_dir.parent / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = list(subtracted_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {subtracted_dir}")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, index_col=0)

        results = []

        # Precompute base-channel map for columns
        col_base_map = {
            col: _base_channel(col)
            for col in df.columns
        }

        for row_label in df.index:
            row_base = _base_channel(row_label)

            # Find matching column(s)
            matching_cols = [
                col for col, base in col_base_map.items()
                if base == row_base
            ]

            if len(matching_cols) == 0:
                raise KeyError(
                    f"No matching column for row '{row_label}' "
                    f"in {csv_path.name}"
                )

            if len(matching_cols) > 1:
                raise ValueError(
                    f"Multiple matching columns for row '{row_label}' "
                    f"in {csv_path.name}: {matching_cols}"
                )

            col_label = matching_cols[0]
            value = df.loc[row_label, col_label]

            results.append({
                "channel": row_label,
                "base_channel": row_base,
                "value": value
            })

        out_df = pd.DataFrame(results)

        output_path = output_dir / (
            csv_path.stem + "_similarity_to_self.csv"
        )
        out_df.to_csv(output_path, index=False)

        print(f"Saved similarity-to-self: {output_path}")
