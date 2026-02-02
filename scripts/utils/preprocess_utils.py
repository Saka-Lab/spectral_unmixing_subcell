# Standard library imports
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import torch
# import einops
import numpy as np
import pandas as pd
from bioio import BioImage
from skimage import measure

# Classes
from skimage.measure._regionprops import RegionProperties



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