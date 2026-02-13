import pandas as pd
from pathlib import Path
import re

def extract_entity(label: str) -> str:
    """
    Extract entity name from a label like:
    'DAPI - Gr1' -> 'Gr1'
    'DAPI - full' -> 'full'
    """
    return label.split(" - ")[1]

def normalize_matrix(
    csv_path: Path,
    divisors_ref: dict,
    divisors_group: dict,
    output_dir: Path
):
    """
    Normalize a single GT-organized PCC matrix.
    """

    df = pd.read_csv(csv_path, index_col=0)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = csv_path.name

    # Determine matrix type from filename
    if "_GT_vs_GT_" in filename:
        matrix_type = "group"
    else:
        matrix_type = "reference"

    normalized = df.copy().astype(float)

    for row in df.index:
        row_entity = extract_entity(row)

        for col in df.columns:
            col_entity = extract_entity(col)

            value = df.loc[row, col]

            # ----------------------------
            # Select divisor
            # ----------------------------
            if matrix_type == "reference":
                if row_entity not in divisors_ref:
                    raise KeyError(f"Missing reference divisor for {row_entity}")

                if col_entity not in divisors_ref[row_entity]:
                    raise KeyError(
                        f"Missing divisor for {row_entity} vs {col_entity}"
                    )

                divisor = divisors_ref[row_entity][col_entity]

            else:  # GT vs GT
                if row_entity not in divisors_group:
                    raise KeyError(f"Missing group divisor for {row_entity}")

                if col_entity not in divisors_group[row_entity]:
                    raise KeyError(
                        f"Missing divisor for {row_entity} vs {col_entity}"
                    )

                divisor = divisors_group[row_entity][col_entity]

            # ----------------------------
            # Normalize
            # ----------------------------
            normalized.loc[row, col] = value / divisor

    # ----------------------------
    # Save
    # ----------------------------
    output_path = output_dir / filename.replace(
        "_gb_ncc_matrix.csv",
        "_DAPI_normalized.csv"
    )

    normalized.to_csv(output_path)

    return normalized

def normalize_all_matrices(
    gt_dir,
    divisors_ref,
    divisors_group,
    output_subdir="DAPI_normalized"
):
    """
    Normalize all GT matrices in a directory.
    """

    gt_dir = Path(gt_dir)
    output_dir = gt_dir.parent / output_subdir

    for csv_path in gt_dir.glob("*.csv"):
        normalize_matrix(
            csv_path,
            divisors_ref,
            divisors_group,
            output_dir
        )
