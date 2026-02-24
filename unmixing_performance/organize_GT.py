from pathlib import Path
import pandas as pd

def build_gt_from_reference_single_fov(
    matrix_dir,
    fov,
    reference,   # str OR list[str]
    desired_column_order
):
    """
    Build FOVX_<reference>_vs_GT_gb_ncc_matrix.csv
    for one or more references ("raw", "group", "full").
    """

    matrix_dir = Path(matrix_dir)

    # Allow both str and list[str]
    if isinstance(reference, str):
        references = [reference]
    else:
        references = list(reference)

    outputs = {}

    for ref in references:
        # -------------------------------
        # Load ref vs GrY matrices
        # -------------------------------
        matrices = {}

        for csv_path in matrix_dir.glob(
            f"{fov}_{ref}_vs_Gr*_gb_ncc_matrix.csv"
        ):
            gr = csv_path.stem.split("_vs_")[1].split("_")[0]
            matrices[gr] = pd.read_csv(csv_path, index_col=0)

        if not matrices:
            raise ValueError(
                f"No matrices found for {fov}_{ref}_vs_GrX"
            )

        # -------------------------------
        # Build GT matrix
        # -------------------------------
        base_df = next(iter(matrices.values()))
        gt_df = pd.DataFrame(index=base_df.index)

        for col in desired_column_order:
            gr = col.split(" - ")[1]

            if gr not in matrices:
                raise KeyError(f"[{ref}] Missing matrix for {gr}")

            if col not in matrices[gr].columns:
                raise KeyError(
                    f"[{ref}] Column '{col}' missing in vs {gr}"
                )

            gt_df[col] = matrices[gr][col]

        # -------------------------------
        # Save output
        # -------------------------------
        output_path = matrix_dir / (
            f"{fov}_{ref}_vs_GT_gb_ncc_matrix.csv"
        )
        gt_df.to_csv(output_path)

        outputs[ref] = gt_df

    return outputs

def build_gt_from_groups_single_fov(
    matrix_dir,
    fov,
    desired_row_order,
    desired_column_order
):
    """
    Build FOVX_GT_vs_GT_gb_ncc_matrix.csv
    from FOVX_GrX_vs_GrY_gb_ncc_matrix.csv files.
    """

    matrix_dir = Path(matrix_dir)

    # -------------------------------
    # Load GrX vs GrY matrices
    # -------------------------------
    matrices = {}

    for csv_path in matrix_dir.glob(
        f"{fov}_Gr*_vs_Gr*_gb_ncc_matrix.csv"
    ):
        parts = csv_path.stem.split("_")
        left = parts[1]   # GrX
        right = parts[3]  # GrY

        matrices.setdefault(left, {})[right] = (
            pd.read_csv(csv_path, index_col=0)
        )

    if not matrices:
        raise ValueError(
            f"No group comparison matrices found for {fov}"
        )

    # -------------------------------
    # Build GT matrix (rows + cols)
    # -------------------------------
    example_df = next(iter(next(iter(matrices.values())).values()))
    gt_df = pd.DataFrame(
        index=desired_row_order,
        columns=desired_column_order,
        dtype=float
    )

    for row_label in desired_row_order:
        gr_row = row_label.split(" - ")[1]

        for col_label in desired_column_order:
            gr_col = col_label.split(" - ")[1]

            if gr_row not in matrices or gr_col not in matrices[gr_row]:
                raise KeyError(f"Missing matrix {gr_row} vs {gr_col}")

            src_df = matrices[gr_row][gr_col]

            if row_label not in src_df.index:
                raise KeyError(f"Row '{row_label}' missing in {gr_row} vs {gr_col}")

            if col_label not in src_df.columns:
                raise KeyError(f"Column '{col_label}' missing in {gr_row} vs {gr_col}")

            gt_df.loc[row_label, col_label] = (
                src_df.loc[row_label, col_label]
            )

    # -------------------------------
    # Save output
    # -------------------------------
    output_path = matrix_dir / (
        f"{fov}_GT_vs_GT_gb_ncc_matrix.csv"
    )
    gt_df.to_csv(output_path)

    return gt_df

def build_gt_matrices_single_fov(
    pcc_dir,
    fov,
    desired_row_order,
    desired_column_order,
    references=("raw", "group", "full"),
    output_subdir="gt_matrices"
):
    """
    Step 2: build all GT matrices for a single FOV.
    """

    pcc_dir = Path(pcc_dir)
    fov_dir = pcc_dir.parent
    output_dir = fov_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------
    # Reference vs GT (raw/group/full)
    # ---------------------------------
    ref_outputs = build_gt_from_reference_single_fov(
        matrix_dir=pcc_dir,
        fov=fov,
        reference=references,
        desired_column_order=desired_column_order
    )

    # Move outputs to gt_matrices/
    for ref in ref_outputs:
        src = pcc_dir / f"{fov}_{ref}_vs_GT_gb_ncc_matrix.csv"
        dst = output_dir / src.name
        src.replace(dst)

    # ---------------------------------
    # GT vs GT
    # ---------------------------------
    gt_df = build_gt_from_groups_single_fov(
        matrix_dir=pcc_dir,
        fov=fov,
        desired_row_order=desired_row_order,
        desired_column_order=desired_column_order
    )

    src = pcc_dir / f"{fov}_GT_vs_GT_gb_ncc_matrix.csv"
    dst = output_dir / src.name
    src.replace(dst)

    return {
        "GT_vs_GT": gt_df,
        **{f"{r}_vs_GT": df for r, df in ref_outputs.items()}
    }

