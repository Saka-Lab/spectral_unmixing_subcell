from pathlib import Path
import typer
from utils.preprocess_utils import get_experiments
from utils.experiment import Preprocess
from loguru import logger
from tqdm import tqdm
from datetime import datetime

logger.add(f"logs/preprocessing/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
app = typer.Typer(help="Process and run Subcell on multiplexed images.")

CHANNELS_15PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
CHANNELS_14PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
SEG_CHANNELS = ['DAPI', 'VIMENTIN', 'WGA', 'alphaTUBULIN']


def configure_experiment_channels(experiment):
    plex14 = {'round_1_gt', 'round_1_UMX'}
    data_2D = {'round_1_gt', 'round_1_UMX', 'round_1_UMX_branched', 'round_1_gt_branched'}

    channels = CHANNELS_14PLEX if experiment in plex14 else CHANNELS_15PLEX
    n_dim = 2 if experiment in data_2D else 3

    return channels, n_dim


@app.command()
def run(
    input_dir: Path = typer.Option(
        Path("subcell_data/experiments"),
        "--input-dir",
        "-i",
        help="Directory containing subdirectories of experiments that contain single FOV image files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    tmp_dir: Path = typer.Option(
        Path("subcell_data/tmp"),
        "--tmp-dir",
        "-t",
        help="Temporary directory for intermediate results",
    ),
    output_dir: Path = typer.Option(
        Path("subcell_data/preprocessing_results"),
        "--output-dir",
        "-o",
        help="Directory to save the processed images and segmentations",
        exists=False,
        file_okay=False,
        dir_okay=True,
    ),
    min_th: float = typer.Option(
        10.00,
        "--min-th",
        "-mi",
        help="Minimum threshold for percentile normalization",
    ),
    max_th: float = typer.Option(
        99.99,
        "--max-th",
        "-ma",
        help="Maximum threshold for percentile normalization",
    ),
    seg_model: str = typer.Option(
        "vit_b_lm",
        "--seg-model",
        "-s",
        help="Type of model to use for segmentation",
    ),
    min_cell_area: int = typer.Option(
        1500,
        "--min-cell-area",
        help="Minimum area of a cell to be considered valid",
    ),
):
    logger.info("Running with arguments:")
    for key, value in locals().items():
        typer.echo(f"\t{key}: {value}")
    typer.echo("-" * 180)
    typer.echo()

    experiments = get_experiments(input_dir)

    for experiment in tqdm(experiments, total=len(experiments)):
        channels, n_dim = configure_experiment_channels(experiment.name)

        logger.info(
            f"Processing {experiment.name} "
            f"with {len(channels)} channels and {n_dim} dimensions."
        )

        exp = Preprocess(
            name=experiment.name,
            exp_dir=experiment,
            channels=channels,
            min_th=min_th,
            max_th=max_th,
            seg_channels=SEG_CHANNELS,
            seg_model=seg_model,
            n_dim=n_dim,
            min_cell_area=min_cell_area,
            tmp_dir=tmp_dir,
            res_dir=output_dir,
        )
        exp.process_images()


if __name__ == "__main__":
    app()