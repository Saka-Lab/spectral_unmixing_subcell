import argparse
from bioio import BioImage
import bioio_lif
from bioio.writers import OmeTiffWriter
from pathlib import Path

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


def split_lif(experiments, input_dir, output_dir, channels):
    print(f'\nSeparating scenes...')
    
    for exp in experiments:
        print(f'\nExperiment: {exp}')
        exp_dir = input_dir / exp
        images = exp_dir.iterdir()
        images = [image for image in images if image.suffix.lower() in ('.tiff', '.tif', '.lif')]
        print(f"Found {len(images)} files in {exp}.")

        out_dir = output_dir / exp
        # check if the output directory exists, if not create it
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            if image.suffix.lower() == '.lif':
                img = BioImage(image, reader=bioio_lif.Reader)
            else:
                img = BioImage(image)

            scenes = sorted(img.scenes)
            if len(scenes) > 1:
                print(f"  Found {len(scenes)} scenes in {image}.")

            for j, scene in enumerate(scenes):
                scene_name = get_scene_name(image, img, scene, j)
                scene_file = out_dir / f'{scene_name}.tif'

                img.set_scene(scene)
                print(f"  S {j}: {scene_name} - {img.shape}")

                # get the image data
                img_data = img.data     # Opened as TCZYX

                if img_data.shape[0] != 1:
                    raise NotImplementedError("Time dimension is not supported.")
                img_data = img_data[0]  # Remove T dimension -> CZYX

                # if a projection has already been done along Z, the shape will be C1YX, some images may have 1CYX
                if img_data.shape[0] == 1:
                    img_data = img_data[0]  # Remove Z dimension -> CYX
                    dims = 'CYX'
                elif img_data.shape[1] == 1:
                    img_data = img_data[:, 0, :, :]  # Remove Z dimension -> CYX
                    dims = 'CYX'
                else:
                    dims = 'CZYX'

                print(f"Image data shape after removing T dimension: {img_data.shape}, dims: {dims}")

                # check if the image has the correct number of channels
                if img_data.shape[0] != len(channels):
                    raise ValueError(f"Image has {img_data.shape[0]} channels, but expected {len(channels)} channels.")
                print(f"Saving scene {j} with shape {img_data.shape} to {scene_file}")

                # Save the image
                OmeTiffWriter.save(
                    img_data,
                    scene_file,
                    dim_order=dims,  # must match your array shape
                    channel_names=channels,
                    image_name=scene_name,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split LIF files")
    parser.add_argument("--input_dir", help="Path to the input LIF file", type=str, default='raw')
    parser.add_argument("--experiment", help="Experiment name to filter files", type=str, default=None)
    parser.add_argument("--output_dir", help="Directory to save the split files", type=str, default='experiments')
    parser.add_argument("--channels", help="List of channel names", type=str, nargs='+', default=["DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # get the list of experiments
    experiments = [args.experiment] if args.experiment else [p.name for p in input_dir.iterdir()]
    # check that all elements in experiments are directories
    experiments = [exp for exp in experiments if (input_dir / exp).is_dir()]
    # split the files for each experiment
    split_lif(experiments, input_dir, output_dir, args.channels)
