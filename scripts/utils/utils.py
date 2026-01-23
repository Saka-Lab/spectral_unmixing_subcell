# Standard library imports
import yaml
import shutil
import subprocess
import requests
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import boto3
import torch
import einops
import numpy as np
import pandas as pd
from bioio import BioImage
from skimage import measure
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

from urllib.parse import urlparse

# Classes
from skimage.measure._regionprops import RegionProperties
from utils.vit_model import ViTPoolClassifier



@dataclass
class Spec3D:
    x: int
    y: int
    z: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def tuple(self):
        return self.x, self.y, self.z

    @classmethod
    def from_tuple(cls, t):
        return cls(*t)

class SerializableRegionProperties(RegionProperties):
    def __init__(self, slice=None, label=None, label_image=None, intensity_image=None, cache_active=None, extra_properties=None, original_crop=None):
        super().__init__(slice, label, label_image, intensity_image, cache_active, extra_properties=extra_properties)
        self.original_crop = original_crop
        self._initialized = True

    def __getattr__(self, attr):
        if attr == "_initialized" or not self._initialized:
            self.__init__()

        super().__getattr__(attr)

    @classmethod
    def from_regionprops(cls, regionprops, original_crop=None):
        return cls(
            slice=regionprops.slice,
            label=regionprops.label,
            label_image=regionprops._label_image,
            intensity_image=regionprops._intensity_image,
            cache_active=regionprops._cache_active,
            extra_properties=regionprops._extra_properties,
            original_crop=original_crop
        )


def get_experiments(input_dir):
    experiments = input_dir.iterdir()
    # filter out files, keep only directories
    experiments = [exp for exp in experiments if exp.is_dir()]
    # sort experiments by name
    experiments = sorted(experiments)
    if not experiments:
        raise ValueError(f"No experiments found in {input_dir}. Please check the directory structure.")
    print(f"Found {len(experiments)} experiments in {input_dir}.")
    return experiments


def repack_h5_file(input_path, remove_original=True):
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + '_repacked.h5')

    if not shutil.which("h5repack"):
        raise RuntimeError("h5repack not found — install it first (e.g., via `apt install hdf5-tools`).")

    cmd = ['h5repack', str(input_path), str(output_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("h5repack error output:")
        print(result.stderr)
        raise RuntimeError("h5repack failed")

    if remove_original:
        input_path.unlink()
        output_path.rename(input_path)


def protein_image_generator(full_img, channel_map, protein_channels):
    """
    Given a full image and a mapping of channel names to indices,
    returns a generator that yields images for each protein of interest.

    Per protein of interest 4 images are returned:
        1. - 3. the reference channels
        4. the protein of interest

    For each 4 image tuple the name of the protein of interest is also returned
    to be used as prefix for later analysis.
    """

    ref_channels = list(channel_map.values())

    for protein_name, protein_channel in protein_channels.items():
        img_channels = ref_channels + [protein_channel]
        selected_channels = torch.stack([full_img[i] for i in img_channels], dim=0)
        yield selected_channels, protein_name


def check_img_dims(img, patch_size):
    assert len(img.shape) == 4, f"Expected 4D tensor, got {len(img.shape)}D tensor"
    return einops.reduce(img, "c x y z -> c x y", "sum", z=1, x=patch_size.x, y=patch_size.y)


def preprocess_tensor(
    img,
    tile_spec,
    lower_percentile=1,
    upper_percentile=99,
    channel_str=None,
    downsample_kernel=None,
    downsampling_method="mean",
    clamp=True,
):
    """
    Clamps and scales each channel of the input tensor based on specified percentiles.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape (C, Z, Y, X).
    - lower_percentile (float): Lower percentile for clamping (default: 1).
    - upper_percentile (float): Upper percentile for clamping (default: 99).

    Returns:
    - torch.Tensor: Processed tensor with values scaled from 0 to 1 and dtype torch.float32.
    """

    # Ensure tensor is of type float32
    if isinstance(img, torch.Tensor):
        img = img.float()
    else:
        img = torch.tensor(img, dtype=torch.float32)

    if channel_str is not None:
        img = einops.rearrange(img, f"{' '.join(channel_str)} -> C Z Y X")

    # downsample kernel is 1 1 8 --> no downsampling on x and y, downsample on z
    if downsample_kernel is not None:
        if "z-stack" in downsampling_method:
            level = downsampling_method.split(':')[1]
            img = img[:, int(level):int(level)+1, :, :]
        else:
            img = einops.reduce(
                img, "C (z z1) (y y1) (x x1) -> C z y x",
                downsampling_method,
                x1=downsample_kernel.x,
                y1=downsample_kernel.y,
                z1=downsample_kernel.z,
            )

        tile_spec = Spec3D(
            int(tile_spec.x / downsample_kernel.x),
            int(tile_spec.y / downsample_kernel.y),
            int(tile_spec.z / downsample_kernel.z),
        )


    # Flatten spatial dimensions for percentile computation
    img = einops.rearrange(img, "C Z Y X -> (Z Y X) C", Z=tile_spec.z, Y=tile_spec.y, X=tile_spec.x)

    # Compute lower and upper percentile values for each channel
    lower_values = torch.quantile(img, lower_percentile / 100.0, dim=0) 
    upper_values = torch.quantile(img, upper_percentile / 100.0, dim=0)

    if clamp:
        # Clamp the tensor based on computed percentiles
        img = torch.clamp(img, min=lower_values, max=upper_values) # <lower_percentile set to lower_percentile and >upper_percentile set to upper_percentile
    else:
        img = torch.where(img < lower_values, 0, img)
        img = torch.where(img > upper_values, upper_values, img)

    # Avoid division by zero in case upper and lower values are equal
    denominator = upper_values - lower_values
    denominator[denominator == 0] = 1  # Set denominator to 1 where upper == lower
    # Scale the clamped tensor to range [0, 1]
    img = (img - lower_values) / denominator
    # rearrange dims to C X Y Z
    img = einops.rearrange(img, "(Z Y X) C -> C X Y Z", Z=tile_spec.z, Y=tile_spec.y, X=tile_spec.x)

    # check if all channels are 0 to 1 scaled
    for channel in img:
        assert channel.min() == 0 and (channel.max() == 1 or channel.max() == 0), f"Channel {channel} not in range [0, 1], got {channel.min()} to {channel.max()}"

    return img

def load_img_data(img_path, axis_str):
    """Load image data from file."""
    img_data = BioImage(img_path).data
    # AICSImage returns TCZYX
    if axis_str == 'CZYX':
        img_data = img_data[0, :]
    elif axis_str == 'ZYX':
        img_data = img_data[0, 0, :]
    return img_data


def get_regionprops(img, labels, img_axis_str, label_axis_str):
    """Get region properties from labels."""
    # ensure correct dimensions
    img = einops.rearrange(img, f"{' '.join(img_axis_str)} -> X Y Z C")
    labels = einops.rearrange(labels, f"{' '.join(label_axis_str)} -> X Y Z")

    # get region props
    raw_props = measure.regionprops(labels, intensity_image=img)
    regionprops = []
    # I need the cropped image and the mask
    for regionprop in raw_props:
        min_x, min_y, min_z, max_x, max_y, max_z = regionprop.bbox
        original_crop = img[min_x:max_x, min_y:max_y, min_z:max_z]
        serializable = SerializableRegionProperties.from_regionprops(regionprops=regionprop, original_crop=original_crop)
        regionprops.append(serializable)

    return regionprops

def get_patch(regionprop, patch, patch_size, bg_masking):
    if bg_masking:
        img = regionprop.image_intensity
        mask = einops.repeat(regionprop.image.astype(int), "X Y Z -> X Y Z C", C=img.shape[-1])
        img = img * mask
    else:
        img = regionprop.original_crop

    pad_img = np.zeros((
            img.shape[0] + patch_size.x,
            img.shape[1] + patch_size.y,
            img.shape[2] + patch_size.z,
            img.shape[3])
    )
    pad_img[
        patch_size.x // 2 : -patch_size.x // 2,
        patch_size.y // 2 : -patch_size.y // 2,
        patch_size.z // 2 : -patch_size.z // 2,
        :,
    ] = img

    img_patch = pad_img[
        patch.x : patch.x + patch_size.x,
        patch.y : patch.y + patch_size.y,
        patch.z : patch.z + patch_size.z,
        :,
    ]

    return img_patch


def get_model(config):
    # We load the selected model information
    model_dir = Path("models") / config["model_channels"] / config["model_type"]
    with open(model_dir / "model_config.yaml", "r") as config_buffer:
        model_config_file = yaml.safe_load(config_buffer)

    classifier_paths = model_config_file["classifier_paths"]
    encoder_path = model_config_file["encoder_path"]

    needs_update = any(not Path(p).is_file() for p in classifier_paths)
    needs_update = needs_update or not Path(encoder_path).is_file()

    # Checking for model update
    if needs_update:
        config["log"].info("- Downloading models...")
        with open("models_urls.yaml", "r") as urls_file:
            url_info = yaml.safe_load(urls_file)
            for index, curr_url_info in enumerate(url_info[config["model_channels"]][config["model_type"]]["classifiers"]):
                if curr_url_info.startswith("s3://"):
                    try:
                        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                        urlcomponents = urlparse(curr_url_info)
                        s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], classifier_paths[index])
                        config["log"].info("  - " + classifier_paths[index] + " updated.")
                    except ClientError as e:
                        config["log"].warning("  - " + classifier_paths[index] + " s3 url " + curr_url_info + " not working.")
                else:
                    response = requests.get(curr_url_info)
                    if response.status_code == 200:
                        with open(classifier_paths[index], "wb") as downloaded_file:
                            downloaded_file.write(response.content)
                        config["log"].info("  - " + classifier_paths[index] + " updated.")
                    else:
                        config["log"].warning("  - " + classifier_paths[index] + " url " + curr_url_info + " not found.")

            curr_url_info = url_info[config["model_channels"]][config["model_type"]]["encoder"]
            if curr_url_info.startswith("s3://"):
                try:
                    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    urlcomponents = urlparse(curr_url_info)
                    s3.download_file(urlcomponents.netloc, urlcomponents.path[1:], encoder_path)
                    config["log"].info("  - " + encoder_path + " updated.")
                except ClientError as e:
                    config["log"].warning("  - " + encoder_path + " s3 url " + curr_url_info + " not working.")
            else:
                response = requests.get(curr_url_info)
                if response.status_code == 200:
                    with open(encoder_path, "wb") as downloaded_file:
                        downloaded_file.write(response.content)
                    config["log"].info("  - " + encoder_path + " updated.")
                else:
                    config["log"].warning("  - " + encoder_path + " url " + curr_url_info + " not found.")

    model_config = model_config_file.get("model_config")
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_path, classifier_paths)
    return model, classifier_paths


def parse_id(id_str):

    parts = id_str.split('_')

    cell_id = parts[-2]
    protein = parts[-1]
    image_name = '_'.join(parts[0:-2])
    unique_cell_id = f"{image_name}_{cell_id}"
    condition = get_condition(image_name)

    return {
        'condition': condition,
        'cell_id': cell_id,
        'protein': protein,
        'image_name': image_name,
        'unique_cell_id': unique_cell_id
    }


def get_condition(im_name):
    """
    This function will have to be adapted based on the experiment naming conventions.
    """

    # List of condition names and their possible keywords in the image name
    actD = ['ActD', 'Act D', 'ActinomycinD', 'Actinomycin D']
    sodium_arsenite = ['Sodium Arsenite', 'SodiumArsenite', 'NaAsO2', 'NaAsO']
    control = ['Control', 'control', 'Unperturbed', 'unperturbed', 'Untreated', 'untreated']

    conditions_map = {
        'ActD': actD,
        'SodiumArsenite': sodium_arsenite,
        'Control': control
    }

    for condition, keywords in conditions_map.items():
        for keyword in keywords:
            if keyword in im_name:
                return condition
    return 'Unknown'


def load_data(input_dir, annotations_dir, round_name, model, interphase_only=False, rename_map=None):
    data_file = Path(input_dir) / f"{round_name}_{model}.csv"
    annotation_file = Path(annotations_dir) / f"{round_name}_manual.csv"
    # Read data
    # check if the file exists
    if data_file.exists():
        df = pd.read_csv(data_file, sep="\t")
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")
    # check if annotations exist
    if annotation_file.exists():
        annotations = pd.read_csv(annotation_file, sep="\t")
        df = pd.merge(df, annotations, on="unique_cell_id", how="left")
        if interphase_only:
            df = df[df["cell_cycle_phase"] == "Interphase"]

    # Features columns are not needed
    feat_cols = [col for col in df.columns if 'feat' in col]
    df = df.drop(columns=feat_cols)
    # Sigmoid probabilities not needed
    sigmoid_cols = [col for col in df.columns if 'sig' in col]
    df = df.drop(columns=sigmoid_cols)

    # Classification from Subcell is not needed
    class_cols = ['classification', 'id', 'top_class_name', 'top_class', 'top_3_classes_names', 'top_3_classes']
    df = df.drop(columns=class_cols)

    if rename_map is not None:
        df = df.rename(columns=rename_map)
    return df