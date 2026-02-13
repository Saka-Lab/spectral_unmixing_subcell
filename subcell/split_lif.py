import typer
from bioio import BioImage
import bioio_lif
from bioio.writers import OmeTiffWriter
from pathlib import Path
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Split LIF/TIFF files into per-scene OME-TIFFs.")


def get_scene_name(image, img, scene, scene_index):
    if image.suffix.lower() in ('.tiff', '.tif'):  
        try:
            scene_name = img.metadata.images[scene_index].name
        except AttributeError:
            scene_name = image.stem
    else:
        scene_name = scene
    scene_name = scene_name.replace(' ', '_').replace('/', '_')
    return scene_name


def split_lif(experiments, input_dir, output_dir, channels, scene_filter):
    logger.info(f"Splitting image files of {len(experiments)} experiments.")
    for exp in experiments:
        logger.info(f'Splitting image file of experiment: {exp}')
        exp_dir = input_dir / exp
        images = exp_dir.iterdir()
        images = [image for image in images if image.suffix.lower() in ('.tiff', '.tif', '.lif')]
        logger.info(f"Found {len(images)} files in {exp}.")

        out_dir = output_dir / exp
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            if image.suffix.lower() == '.lif':
                img = BioImage(image, reader=bioio_lif.Reader)
            else:
                img = BioImage(image)

            scenes = sorted(img.scenes)
            if len(scenes) > 1:
                logger.info(f"  Found {len(scenes)} scenes in {image}.")

            for index, scene in tqdm(enumerate(scenes), total=len(scenes)):
                scene_name = get_scene_name(image, img, scene, index)
                if scene_filter and scene_filter not in scene_name:
                    logger.info(f"Skipping scene as it does not contain scene filter `{scene_filter}`: {scene_name}")
                    continue
                scene_file = out_dir / f'{scene_name}.tif'

                img.set_scene(scene)
                logger.info(f"  S {index}: {scene_name} - {img.shape}")

                img_data = img.get_image_data("CZYX")
                if img_data.shape[1] == 1:
                    img_data = img_data[:, 0, :, :]  # Remove Z dimension -> CYX
                    dims = 'CYX'
                else:
                    dims = 'CZYX'

                if img_data.shape[0] != len(channels):
                    raise ValueError(f"Image has {img_data.shape[0]} channels, but  {len(channels)} channels were defined.")
                logger.info(f"Saving scene {index} with shape {img_data.shape} to {scene_file}")

                OmeTiffWriter.save(
                    img_data,
                    scene_file,
                    dim_order=dims,
                    channel_names=channels,
                    image_name=scene_name,
                )


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("subcell_data/raw"),
        "--input-dir",
        "-i",
        help="Path to the input directory containing experiments. (default: data/raw)"
    ),
    experiments: list[str] = typer.Option(
        [],
        "--experiments",
        "-e",
        help="Experiment name to process (default: all subdirectories)."
    ),
    output_dir: Path = typer.Option(
        Path("subcell_data/experiments"),
        "--output-dir",
        "-o",
        help="Path to the output directory where resulting images are stored. (default: data/experiments)",

    ),
    channels: list[str] = typer.Option(
        [
            "DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON",
            "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM",
            "Ki67", "NPM1"
        ],
        "--channels",
        "-c",
        help="List of channel names. Length should match with the number of channels in the images."
    ),
    scene_filter: str = typer.Option(
        "ST",
        "--scene_filter",
        "-sf",
        help="String based on which to filter images in a lif file.",
    ),
):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not experiments:
        experiments = [p.name for p in input_dir.iterdir() if p.is_dir()]

    split_lif(experiments, input_dir, output_dir, channels, scene_filter)


if __name__ == "__main__":
    app()
