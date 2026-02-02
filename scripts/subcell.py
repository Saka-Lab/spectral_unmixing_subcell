import argparse
import json
from functools import partial
from pathlib import Path

from utils.utils import get_experiments, preprocess_tensor
from utils.experiment import Subcell
from utils.utils import Spec3D



CHANNELS_15PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SC35", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
CHANNELS_14PLEX = ["DAPI", "TOMM20", "alphaTUBULIN", "SP100", "WGA", "SON", "VIMENTIN", "LAMP1", "COILIN", "GM130", "G3BP1", "TFAM", "Ki67", "NPM1"]
ref_channels = ['DAPI', 'alphaTUBULIN', "ER"]

def configure_experiment_channels(experiment):
    plex14 = {'round_1_gt', 'round_1_UMX'}
    data_2D = {'round_1_gt', 'round_1_UMX', 'round_1_UMX_branched', 'round_1_gt_branched'}

    channels = CHANNELS_14PLEX if experiment in plex14 else CHANNELS_15PLEX
    n_dim = 2 if experiment in data_2D else 3
    return channels, n_dim

def main(args):
    experiments = get_experiments(Path(args.input_dir)/"8bits"/f"{args.min_th}_{args.max_th}")
    experiments = [exp.name for exp in experiments]

    cell_patch_size = Spec3D(**args.cell_patch_size)
    downsample_kernel = Spec3D(**args.downsample_kernel)
    transform = partial(
        preprocess_tensor,
        tile_spec=cell_patch_size,                       # size of the cell patches
        channel_str=args.channel_order,                  # order of channels of input image
        downsample_kernel=downsample_kernel,             # aggregate over z axis
        downsampling_method=args.downsampling_method,    # method to downsample
    )

    for experiment in experiments:
        args.channels, _ = configure_experiment_channels(experiment)
        
        print(f"Processing experiment: {experiment}")
        subcell = Subcell(
            name=experiment,
            min_th=args.min_th,
            max_th=args.max_th,
            transform=transform,
            cell_patch_size=cell_patch_size,
            channels=args.channels,
            ref_channels=args.ref_channels,
            model_channels=args.model_channels,
            input_dir=Path(args.input_dir),
            prep_dir=Path(args.prep_dir),
            res_dir=Path(args.subcell_res_dir),
            gpu=args.gpu
        )

        print(subcell)
        subcell.run_prep(bg_masking=args.bg_masking)
        subcell.run_subcell(bg_masking=args.bg_masking, model_type=args.sc_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subcell analysis")
    parser.add_argument('--input_dir', type=str, help='Path to the experiment directory', default='preprocessing_results')
    parser.add_argument('--cell_patch_size', type=json.loads, help='cell patch size', default='{"x": 256, "y": 256, "z": 8}')
    parser.add_argument('--downsample_kernel', type=json.loads, help='downsample kernel size', default='{"x": 1, "y": 1, "z": 8}')
    parser.add_argument('--channel_order', type=str, help='order of channels in the input image', default='XYZC')
    parser.add_argument("--model_channels", help="channel images to be used [rybg, rbg, ybg, bg]", default="rbg", type=str)
    parser.add_argument('--downsampling_method', type=str, help='downsampling method', default='sum', choices=['sum', 'max', 'mean'])
    parser.add_argument('--min_th', type=str, help='minimum threshold for segmentation', default='10.00')
    parser.add_argument('--max_th', type=str, help='maximum threshold for segmentation', default='99.99')
    parser.add_argument('--prep_dir', type=str, help='Path to the subcell results directory', default='subcell_prep')
    parser.add_argument('--subcell_res_dir', type=str, help='Path to the subcell results directory', default='subcell_results')
    parser.add_argument('--gpu', type=int, help='Whether to use the gpu or not', default=0)
    parser.add_argument('--bg_masking', type=bool, help='Whether to use background masking or not', default=False)
    parser.add_argument('--sc_model', type=str, help='Self-supervised model to use', default='mae_contrast_supcon_model', choices=['mae_contrast_supcon_model', 'vit_supcon_model'])

    args = parser.parse_args()
    args.ref_channels = ref_channels

    print(f"Running with args:")
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    print('---'*60)
    print()

    main(args)