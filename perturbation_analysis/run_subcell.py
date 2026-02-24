import json
from functools import partial
from pathlib import Path
import typer
from utils.cellregions import Spec3D
from utils.subcell_utils import preprocess_tensor, Subcell
from utils.preprocess_utils import get_experiments
from enum import Enum
from typing import Annotated
from loguru import logger
from datetime import datetime

logger.add(f"logs/subcell/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


class DownsampleMethod(str, Enum):
    sum = 'sum'
    max = 'max'
    mean = 'mean'


class ModelType(str, Enum):
    mae_contrast_supcon_model = "mae_contrast_supcon_model"
    vit_supcon_model = "vit_supcon_model"


CHANNELS_15PLEX = [
    "DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON",
    "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"
]
CHANNELS_14PLEX = [
    "DAPI", "TOMM20", "alphaTUBULIN", "SP100", "WGA", "SON",
    "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"
]

REF_CHANNELS = ["DAPI", "alphaTUBULIN", "ER"]

app = typer.Typer(add_completion=False)


def configure_experiment_channels(experiment):
    plex14 = {'round_1_gt', 'round_1_UMX'}
    data_2D = {'round_1_gt', 'round_1_UMX', 'round_1_UMX_branched', 'round_1_gt_branched'}

    channels = CHANNELS_14PLEX if experiment in plex14 else CHANNELS_15PLEX
    n_dim = 2 if experiment in data_2D else 3
    return channels, n_dim


@app.command()
def main(
        input_dir: Path = typer.Option(
            "perturbation_data/preprocessing_results",
            "--input-dir",
            "-i",
            help="Directory containing subdirectories of experiments that contain single FOV image files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
        cell_patch_size: str = typer.Option(
            '{"x": 256, "y": 256, "z": 8}',
            "--cell-patch-size",
            "-cp",
            help="Cell patch size",
        ),
        downsample_kernel: str = typer.Option(
            '{"x": 1, "y": 1, "z": 8}',
            "--downsample-kernel",
            "-dk",
            help="Downsample kernel. This is used as input to einops.reduce. If defined, should only define"
                 "downsampling for `x`, `y` and `z` axes. Given an image with z 80 if z is defined as 8, will result"
                 "in z being 10. Downsampling is performed with the defined downsampling method.",
        ),
        model_channels: str = typer.Option(
            "rbg",
            "--model-channels",
            "-mc",
            help="channel images to be used [rybg, rbg, ybg, bg]"),
        downsampling_method: DownsampleMethod = typer.Option(
            "sum",
            "--downsample-method",
            "-ds",
            help="Downsample method",
            show_choices=True,
        ),
        min_th: float = typer.Option(
            10.00,
            "--min-th",
            "-mi",
            help="Minimum threshold for percentile normalization. Purely used for parsing the directory.",
        ),
        max_th: float = typer.Option(
            99.99,
            "--max-th",
            "-ma",
            help="Maximum threshold for percentile normalization. Purely used for parsing the directory.",
        ),
        prep_dir: Path = typer.Option(
            Path("perturbation_data/subcell_prep"),
            "--tmp-dir",
            "-t",
            help="Directory to store temporary files like the individual cell images.",
        ),
        output_dir: Path = typer.Option(
            Path("perturbation_data/subcell_results"),
            "--output-dir",
            "-o",
            help="Output directory of the final results of the SubCell experiment. B"
        ),
        gpu: int = typer.Option(
            0,
            "--gpu",
            "-g",
            help="Index of the GPU device to use (e.g., 0 for first GPU, 1 for second)"
        ),
        bg_masking: Annotated[bool, typer.Option(
            "--bg-masking",
            "-bg",
            help="If given as argument, will apply background masking.",
            is_flag=True)] = False,
        sc_model: ModelType = typer.Option(
            "mae_contrast_supcon_model",
            "--model_type",
            "-mt",
            help="Self-supervised model to use",
            show_choices=True
        ),
):
    logger.info("Running with args:")
    for k, v in locals().items():
        logger.info(f"\t{k}: {v}")
    logger.info("-" * 180)

    cell_patch_size = Spec3D(**json.loads(cell_patch_size))
    downsample_kernel = Spec3D(**json.loads(downsample_kernel))

    experiments = get_experiments(
        Path(input_dir) / "8bits" / f"{min_th}_{max_th}"
    )
    experiments = [exp.name for exp in experiments]

    transform = partial(
        preprocess_tensor,
        tile_spec=cell_patch_size,
        downsample_kernel=downsample_kernel,
        downsampling_method=downsampling_method,
    )

    for experiment in experiments:
        channels, _ = configure_experiment_channels(experiment)

        logger.info(f"Processing experiment: {experiment}")

        subcell = Subcell(
            name=experiment,
            min_th=min_th,
            max_th=max_th,
            transform=transform,
            cell_patch_size=cell_patch_size,
            channels=channels,
            ref_channels=REF_CHANNELS,
            model_channels=model_channels,
            input_dir=Path(input_dir),
            prep_dir=Path(prep_dir),
            res_dir=Path(output_dir),
            gpu=gpu,
        )

        logger.info(str(subcell))
        subcell.run_prep(bg_masking=bg_masking)
        subcell.run_subcell(
            bg_masking=bg_masking,
            model_type=sc_model.name
        )


if __name__ == "__main__":
    app()