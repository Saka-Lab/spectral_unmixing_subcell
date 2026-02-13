import pandas as pd
from pathlib import Path

def sum_rows_single_matrix(
    csv_path: Path,
    output_dir: Path
):
    """
    Sum all values in each row of a matrix.
    """

    df = pd.read_csv(csv_path, index_col=0)

    row_sum_df = pd.DataFrame({
        "row_sum": df.sum(axis=1)
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / csv_path.name.replace(
        "_abs.csv",
        "_row_sum.csv"
    )

    row_sum_df.to_csv(output_path)

    return row_sum_df

def sum_rows_all_matrices(
    subtracted_dir,
    output_subdir="row_sums"
):
    """
    Apply row-wise summation to all subtracted matrices.
    """

    subtracted_dir = Path(subtracted_dir)
    output_dir = subtracted_dir.parent / output_subdir

    for csv_path in subtracted_dir.glob("*.csv"):
        sum_rows_single_matrix(
            csv_path,
            output_dir
        )
