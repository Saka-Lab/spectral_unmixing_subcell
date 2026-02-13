import pandas as pd
from pathlib import Path

def negatives_to_zero_single_matrix(
    csv_path: Path,
    output_dir: Path
):
    """
    Convert negative values in a matrix to zero.
    """

    df = pd.read_csv(csv_path, index_col=0)

    # Apply clipping
    df_clipped = df.clip(lower=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / csv_path.name.replace(
        "_DAPI_normalized.csv",
        "_DAPI_norm_neg0.csv"
    )

    df_clipped.to_csv(output_path)

    return df_clipped

def negatives_to_zero_all(
    normalized_dir,
    output_subdir="neg_to_zero"
):
    """
    Apply negative-to-zero conversion to all DAPI-normalized matrices.
    """

    normalized_dir = Path(normalized_dir)
    output_dir = normalized_dir.parent / output_subdir

    for csv_path in normalized_dir.glob("*_DAPI_normalized.csv"):
        negatives_to_zero_single_matrix(
            csv_path,
            output_dir
        )
