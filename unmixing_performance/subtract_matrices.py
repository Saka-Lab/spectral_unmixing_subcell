import pandas as pd
from pathlib import Path

def subtract_single_reference(
    ref_csv: Path,
    gt_csv: Path,
    output_dir: Path
):
    """
    Compute absolute difference |ref_vs_GT - GT_vs_GT|
    using positional subtraction only.
    """

    ref_df = pd.read_csv(ref_csv, index_col=0)
    gt_df = pd.read_csv(gt_csv, index_col=0)

    # ----------------------------
    # Safety check: shape only
    # ----------------------------
    if ref_df.shape != gt_df.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"{ref_csv.name}: {ref_df.shape}\n"
            f"{gt_csv.name}: {gt_df.shape}"
        )

    # ----------------------------
    # Subtract by position
    # ----------------------------
    diff_values = (ref_df.values - gt_df.values).astype(float)
    diff_df = pd.DataFrame(
        diff_values,
        index=ref_df.index,
        columns=ref_df.columns
    ).abs()

    # ----------------------------
    # Save output
    # ----------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / ref_csv.name.replace(
        "_vs_GT_DAPI_norm_neg0.csv",
        "_minus_GT_abs.csv"
    )

    diff_df.to_csv(output_path)

    return diff_df

def subtract_all_references_single_fov(
    normalized_dir,
    references=("raw", "group", "full"),
    output_subdir="subtracted"
):
    """
    Compute |ref_vs_GT - GT_vs_GT| for all references.
    """

    normalized_dir = Path(normalized_dir)
    fov_name = normalized_dir.parent.name
    output_dir = normalized_dir.parent / output_subdir

    gt_csv = normalized_dir / (
        f"{fov_name}_GT_vs_GT_DAPI_norm_neg0.csv"
    )

    if not gt_csv.exists():
        raise FileNotFoundError(
            f"GT_vs_GT matrix not found: {gt_csv}"
        )

    for ref in references:
        ref_csv = normalized_dir / (
            f"{fov_name}_{ref}_vs_GT_DAPI_norm_neg0.csv"
        )

        if not ref_csv.exists():
            raise FileNotFoundError(
                f"Missing matrix: {ref_csv}"
            )

        subtract_single_reference(
            ref_csv,
            gt_csv,
            output_dir
        )
