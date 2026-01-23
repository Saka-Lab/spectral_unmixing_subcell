# Standard library imports
import argparse
from pathlib import Path

# Module imports
from utils.utils import get_experiments

# Classes 
from utils.experiment import Preprocess


CHANNELS_15PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
CHANNELS_14PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
seg_channels = ['DAPI', 'VIMENTIN', 'WGA', 'alphaTUBULIN']


def parse_args():
    parser = argparse.ArgumentParser(description="Process and run Subcell on multiplexed images.")
    # Directories
    parser.add_argument("--experiment_directory", type=str, default="experiments", help="Directory containing one folder with raw images for each experiment")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Temporary directory for intermediate results")
    parser.add_argument("--prep_res_dir", type=str, default="preprocessing_results", help="Directory to save the processed images and results")
    
    # Preprocessing parameters
    parser.add_argument("--min_th", type=str, default="10.00", help="Minimum threshold for clamping outliers")
    parser.add_argument("--max_th", type=str, default="99.99", help="Maximum threshold for clamping outliers")
    parser.add_argument("--seg_model", type=str, default="vit_b_lm", help="Type of model to use for segmentation")
    parser.add_argument("--min_cell_area", type=int, default=1500, help="Minimum area of a cell to be considered valid")
    parser.add_argument("--min_border_area", type=int, default=40000, help="Minimum area of a cell to be considered valid at the border")

    args = parser.parse_args()
    args.seg_channels = seg_channels
    return args

def configure_experiment_channels(experiment):
    plex14 = {'round_1_gt', 'round_1_UMX'}
    data_2D = {'round_1_gt', 'round_1_UMX', 'round_1_UMX_branched', 'round_1_gt_branched'}

    channels = CHANNELS_14PLEX if experiment in plex14 else CHANNELS_15PLEX
    n_dim = 2 if experiment in data_2D else 3


    return channels, n_dim

def main(args):
    experiments = get_experiments(Path(args.experiment_directory))

    for experiment in experiments:
        # based on the experiment get the channel names and wheteter the data is 2D or 3D
        args.channels, args.n_dim = configure_experiment_channels(experiment.name)

        print(f"\nProcessing {experiment.name} with {len(args.channels)} channels and {args.n_dim} dimensions.")
        exp = Preprocess(
            name=experiment.name, exp_dir=experiment, channels=args.channels,
            min_th=args.min_th, max_th=args.max_th,
            seg_channels=args.seg_channels, seg_model=args.seg_model, n_dim=args.n_dim,
            min_cell_area=args.min_cell_area, min_border_area=args.min_border_area,
            tmp_dir=Path(args.tmp_dir), res_dir=Path(args.prep_res_dir),
        )
        print(f"\nProcessing {exp}")
        exp.process_images()


if __name__ == "__main__":
    args = parse_args()

    print(f"Running with arguments:")
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    print('---'*60)
    print()

    main(args)
    