from pathlib import Path
from pcc import compute_all_pcc_for_single_fov
from organize_GT import build_gt_matrices_single_fov
from divisors import extract_divisors_from_dir
from DAPI_normalizer import normalize_all_matrices
from NegToZero import negatives_to_zero_all
from subtract_matrices import subtract_all_references_single_fov
from row_sum import sum_rows_all_matrices
from aggregate_row_sums import aggregate_and_normalize_row_sums
from plot_row_sums import plot_row_sums_boxplot_with_stats
from similarity_to_self import similarity_to_self_all_matrices
from aggregate_similarity import aggregate_and_normalize_similarity_to_self
from utils_average_matrices import average_matrices_across_fovs
from plot_heatmaps import plot_avg_heatmaps
from loguru import logger
import typer

app = typer.Typer(help="Create pearson correlation heatmaps")

# ----------------------------------
# CONFIGURATION
# ----------------------------------
DESIRED_COLUMN_ORDER = [
    'DAPI - Gr1', 'ALEXA 594 - Gr2', 'ATTO 425 - Gr1',
    'ATTO 633 - Gr3', 'CF770 - Gr3', 'ATTO 488 - Gr1',
    'ATTO 490LS - Gr2', 'Oregon Green 514 - Gr2',
    'ALEXA 532 - Gr3', 'ALEXA 647 - Gr1', 'ATTO 550 - Gr1',
    'TYE 705 - Gr2', 'Atto Rho11 - Gr3', 'ALEXA 750 - Gr2'
]

DESIRED_CHANNEL_ORDER = [
    "DAPI", "ATTO 425", "ATTO 488", "ATTO 490LS", "Oregon Green 514",
    "ALEXA 532", "ATTO 550", "Atto Rho11", "ALEXA 594", "ATTO 633",
    "ALEXA 647", "TYE 705", "ALEXA 750", "CF770"
]

DESIRED_ROW_ORDER = DESIRED_COLUMN_ORDER

REFERENCES = ["raw", "group", "full"]

# ----------------------------------
# PIPELINE
# ----------------------------------
def run_pipeline_for_single_fov(fov_dir: Path):

    logger.info(f"\n=== Processing {fov_dir.name} ===")

    # Step 1: PCC
    compute_all_pcc_for_single_fov(
        fov_dir=fov_dir,
    )

    pcc_dir = fov_dir / "pcc"

    # Step 2: Organize into GT
    build_gt_matrices_single_fov(
        pcc_dir=pcc_dir,
        fov=fov_dir.name,
        references=REFERENCES,
        desired_row_order=DESIRED_ROW_ORDER,
        desired_column_order=DESIRED_COLUMN_ORDER
    )

    # Step 3: Extract divisors
    divisors_ref, divisors_group = extract_divisors_from_dir(pcc_dir)

    gt_dir = fov_dir / "gt_matrices"  # organized GT matrices

    # Step 4: DAPI normalization
    normalize_all_matrices(
        gt_dir=gt_dir,
        divisors_ref=divisors_ref,
        divisors_group=divisors_group
    )

    # Step 5: Negatives → zero
    negatives_to_zero_all(
        normalized_dir=fov_dir / "DAPI_normalized"
    )

    # Step 6: Absolute subtraction
    subtract_all_references_single_fov(
        normalized_dir=fov_dir / "neg_to_zero",
        references=REFERENCES
    )

    # Step 7: Sum rows of subtracted matrices
    sum_rows_all_matrices(
        subtracted_dir=fov_dir / "subtracted"
    )

    # Step 8: Similarity to self
    similarity_to_self_all_matrices(
        subtracted_dir=fov_dir / "subtracted"
    )


def run_pipeline_for_all_fovs(input_dir: Path, experiments: list[str]):

    fov_dirs = [input_dir / exp for exp in experiments]
    for index, fov_dir in enumerate(fov_dirs):
        if not fov_dir.exists():
            logger.debug(f"No directory for experiment {experiments[index]}, {fov_dir} does not exist. Skipping.")
            fov_dirs.pop(index)

    if not fov_dirs:
        raise RuntimeError("No FOV directories found")

    for fov_dir in fov_dirs:
        run_pipeline_for_single_fov(fov_dir)

    # Step 9: Aggregate and normalize row sums across all FOVs
    aggregate_and_normalize_row_sums(
        data_root=input_dir,
        row_sums_subdir="row_sums",
        output_subdir="aggregated_normalized"
    )

    # Step 10: Visualize normalized row sums
    plot_row_sums_boxplot_with_stats(
        normalized_long_csv=input_dir / "aggregated_normalized" / "all_fovs_row_sums_normalized_long.csv",
        title="Normalized Row Sums per Channel",
        output_name="row_sums_boxplot.svg"
    )

    # Step 11: Aggregate & normalize similarity-to-self across all FOVs
    aggregate_and_normalize_similarity_to_self(
        data_root=input_dir,
        similarity_subdir="subtracted_similarity",
        output_subdir="aggregated_similarity"
    )

    # Step 12: Visualize similarity-to-self
    plot_row_sums_boxplot_with_stats(
    normalized_long_csv=input_dir / "aggregated_similarity" / "all_fovs_similarity_to_self_normalized_long.csv",
    title="Similarity to Self (1 - PCC) per Channel",
    output_name="similarity_to_self_boxplot.svg"
    )

    # Step 13: Average DAPI_normalized matrices
    avg_normalized = average_matrices_across_fovs(
        fov_dirs=fov_dirs,
        folder_name="DAPI_normalized",
        comparison_filter=["GT_vs_GT", "raw_vs_GT", "group_vs_GT", "full_vs_GT"],
        filename_regex=r'^FOV\d+_(GT_vs_GT|raw_vs_GT|group_vs_GT|full_vs_GT)_DAPI_normalized\.csv$',
        channel_order=DESIRED_CHANNEL_ORDER
    )
    plot_avg_heatmaps(avg_normalized, output_dir=input_dir / "plots" / "avg_DAPI_normalized")

    # Step 14: Average subtracted matrices
    avg_subtracted = average_matrices_across_fovs(
        fov_dirs=fov_dirs,
        folder_name="subtracted",
        comparison_filter=["raw_minus_GT", "group_minus_GT", "full_minus_GT"],
        filename_regex=r'^FOV\d+_(raw_minus_GT|group_minus_GT|full_minus_GT)_abs\.csv$',
        channel_order=DESIRED_CHANNEL_ORDER
    )
    plot_avg_heatmaps(avg_subtracted, output_dir=input_dir / "plots" / "avg_subtracted")


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("unmixing_performance_data"),
        "--input-dir",
        "-i",
        help="Path to the input directory containing experiments. (default: data/raw)"
    ),
    experiments: list[str] = typer.Option(
        ["FOV1", "FOV2", "FOV3"],
        "--experiments",
        "-e",
        help="Experiment names to process. These must be directory names within `input-dir` and contain ome-tif images with .tif extension."
    ),
):
    run_pipeline_for_all_fovs(input_dir, experiments)


if __name__ == "__main__":
    app()
