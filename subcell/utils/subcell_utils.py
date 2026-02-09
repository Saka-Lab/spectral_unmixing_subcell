import subprocess
from pathlib import Path
import os
import requests
import yaml
from urllib.parse import urlparse
import shutil

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

from utils.vit_model import ViTPoolClassifier
from skimage.io import imread
import datetime
import utils.inference as inference
from tqdm import tqdm
import logging

from .cellregions import CellRegionDataset, Spec3D
import einops

import torchvision
import torch

import h5py



classification = {
    "DAPI":	"Nucleoplasm",
    "TOMM20":	"Mitochondria",
    "alphaTUBULIN":	"Microtubules",
    "SC35":	"Nuclear speckles",
    "SP100":	"Nuclear bodies",
    "WGA":	"Plasma membrane",
    "SON":	"Nuclear speckles",
    "VIMENTIN":	"Intermediate filaments",
    "LAMP1":	"Lysosomes",
    "LAMP2":	"Lysosomes",
    "COILIN":	"Nuclear bodies",
    "GM130":	"Golgi apparatus",
    "G3BP1":	"Cytoplasmic bodies",
    "TFAM":	"Mitochondria",
    "Ki67":	"Nucleoli",
    "NPM1":	"Nucleoli"
}

acceptable_classification = {
    "DAPI":	["Nuclear membrane", "Nucleoli", "Nucleoli rim", "Nuclear speckles", "Nucleoli fibrillar center", "Nuclear bodies"],
    "TOMM20":	[""],
    "alphaTUBULIN":	["Microtubule ends", "Centrosome", "Cytosol"],
    "SC35":	["Nucleoplasm"],
    "SP100":	["Nucleoplasm"],
    "WGA":	["Golgi apparatus", "Nuclear membrane"],
    "SON":	["Nucleoplasm"],
    "VIMENTIN":	["Cytosol", "Focal adhesion sites"],
    "LAMP1":	["Vesicles", "Cytoplasmic bodies"],
    "LAMP2":	["Vesicles", "Cytoplasmic bodies"],
    "COILIN":	["Nucleoplasm"],
    "GM130":	["Vesicles"],
    "G3BP1":	["Cytosol", "Stress granules"],
    "TFAM":	[""],
    "Ki67":	["Nucleoli rim", "Nucleoli fibrillar center", "Nuclear membrane"],
    "NPM1":	["Nucleoli rim", "Nucleoli fibrillar center", "Nuclear speckles"]
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


class Subcell:

    def __init__(self, name, min_th, max_th, transform, cell_patch_size, channels, ref_channels, model_channels,
                 input_dir, prep_dir, res_dir, gpu):
        self.name = name
        self.min_th = min_th
        self.max_th = max_th
        self.transform = transform
        self.cell_patch_size = cell_patch_size
        self.channels = channels
        self.ref_channels = ref_channels
        self.model_channels = model_channels
        self.input_dir = input_dir
        self.prep_dir = prep_dir
        self.res_dir = res_dir
        self.gpu = gpu

    def __repr__(self):
        return f"""Subcell(name={self.name}, min_th={self.min_th}, max_th={self.max_th},
        transform={self.transform},
        cell_patch_size={self.cell_patch_size},
        channels={self.channels},
        ref_channels={self.ref_channels},
        model_channels={self.model_channels},
        input_dir={self.input_dir},
        prep_dir={self.prep_dir},
        res_dir={self.res_dir}"""

    def run_prep(self, bg_masking):
        out_path = Path(f"{self.prep_dir}_bgmask") if bg_masking else self.prep_dir
        out_path = out_path / f"{self.min_th}_{self.max_th}" / self.name
        path_list_file = out_path / f"path_list_{self.name}.csv"
        if path_list_file.is_file():
            df = pd.read_csv(path_list_file)
            if not df.empty:
                print(f"Path list for {out_path} already exists, skipping Subcell preparation.")
                return
        else:
            print(f"File {path_list_file} does not exist.")

        dataset = CellRegionDataset(
            img_dir=self.input_dir / "8bits" / f"{self.min_th}_{self.max_th}",
            labels_dir=self.input_dir / "manual_segmentations",
            patch_size=self.cell_patch_size,
            transform=self.transform,
            h5_dir=out_path,
            bg_masking=bg_masking,
            experiment_name=self.name,
        )

        protein_channel_map = {channel: i for i, channel in enumerate(self.channels)}
        ref_channel_map = {channel: self.channels.index(channel) if channel in self.channels else "" for channel in
                           self.ref_channels}
        ref_channel_keys = [key for key, channel in ref_channel_map.items() if channel != ""]
        selected_channel_map = {key: ref_channel_map[key] for key in ref_channel_map if ref_channel_map[key] != ""}

        img_data = []
        img_path = Path(out_path) / "imgs"
        img_path.mkdir(parents=True, exist_ok=True)

        print("Processing dataset...")
        for idx, (img_name, lab, img) in tqdm(enumerate(dataset), total=len(dataset)):
            img = check_img_dims(img, dataset.patch_size)
            img = img / dataset.patch_size.z
            assert img.min() >= 0 and img.max() <= 1, f"Image values out of range [0, 1]: min {img.min()}, max {img.max()}"
            for protein_img, protein_name in protein_image_generator(img, selected_channel_map, protein_channel_map):
                channel_imgs = []
                p_names = list(selected_channel_map.keys()) + [protein_name]
                for i, img in enumerate(protein_img):
                    p = Path(img_path) / f"{img_name.decode()}_{str(lab)}_{p_names[i]}.png"
                    img = img.unsqueeze(0)
                    torchvision.io.write_png((img * 255).to(torch.uint8), p, compression_level=9)
                    channel_imgs.append(p)

                img_data.append({
                    "r_image": channel_imgs[
                        ref_channel_keys.index("alphaTUBULIN")] if "alphaTUBULIN" in ref_channel_keys else "",
                    "y_image": channel_imgs[ref_channel_keys.index("ER")] if "ER" in ref_channel_keys else "",
                    "b_image": channel_imgs[ref_channel_keys.index("DAPI")] if "DAPI" in ref_channel_keys else "",
                    "g_image": channel_imgs[-1],
                    "prefix": f"{img_name.decode()}_{str(lab)}_{protein_name}",
                }
                )

        dataset.h5f.close()
        df = pd.DataFrame(img_data)
        df.to_csv(path_list_file, index=False, header=False)
        print(f"Saved path list to {path_list_file}")

        # clean up removing the 'cell_regions' key
        with h5py.File(dataset.hdf5_path, 'r+') as f:
            del f['cell_regions']

        # repack the h5 file to reclaim space
        repack_h5_file(dataset.hdf5_path)

    def run_subcell(self, model_type, bg_masking):

        print(f"Running Subcell on experiment: {self.name} - Model: {model_type} - Masking: {bg_masking}")
        paths_file = Path(self.prep_dir) / f"{self.min_th}_{self.max_th}" / self.name / f"path_list_{self.name}.csv"
        results_file = Path(
            self.res_dir) / f"{self.min_th}_{self.max_th}" / f"{self.name}{'_bgmask' if bg_masking else ''}_{model_type.split('_')[0]}.tsv"

        n_classes = 31
        embedding_size = 1536
        os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # This is the log configuration. It will log everything to a file AND the console
        # Check if logs folder exists, if not create it
        os.makedirs("logs", exist_ok=True)

        logger = logging.getLogger(f"SubCell inference-{self.name}-{model_type}-{'bg_masking' if bg_masking else ''}")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent duplicate logs via root logger

        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                f"logs/{self.name}_{model_type}{'_bg_masking' if bg_masking else ''}_log.txt", encoding="utf-8",
                mode="w")
            file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(console_handler)

        # This is the general configuration variable. We are going to use the special key "log" in the dictionary to use the log in our code
        config = {"log": logger}
        config["model_channels"] = self.model_channels
        config["model_type"] = model_type
        config["paths_file"] = paths_file
        config["results_file"] = results_file

        # check if the results directory exists
        res_dir = os.path.dirname(config["results_file"])
        if res_dir:
            os.makedirs(res_dir, exist_ok=True)

        # Log the start time and the final configuration so you can keep track of what you did
        config["log"].info("Start: " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        config["log"].info("Parameters used:")
        config["log"].info(config)
        config["log"].info("----------")

        model, classifier_paths = get_model(config)
        model.eval()

        if torch.cuda.is_available() and self.gpu != -1:
            device = torch.device("cuda:" + str(self.gpu))
            config["log"].warning(f"Using device {device}")
        else:
            config["log"].warning("CUDA not available. Using CPU.")
            device = torch.device("cpu")
        model.to(device)

        final_columns = ["id"]
        if classifier_paths:
            final_columns.extend(["top_class_name", "top_class", "top_3_classes_names", "top_3_classes"])
            prob_columns = []
            for i in range(n_classes):
                prob_columns.append("sig_prob" + "%02d" % (i,))
            for i in range(n_classes):
                prob_columns.append("soft_prob" + "%02d" % (i,))
            final_columns.extend(prob_columns)
            feat_columns = []

        for i in range(embedding_size):
            feat_columns.append("feat" + "%04d" % (i,))
        final_columns.extend(feat_columns)
        df = pd.DataFrame(columns=final_columns)

        # We iterate over each set of images to process
        if not os.path.exists(config["paths_file"]):
            config["log"].error("The paths file does not exist: " + config["paths_file"])
            raise FileNotFoundError(f"The paths file does not exist: {config['paths_file']}")

        with open(config["paths_file"]) as f:
            path_list = f.readlines()
        print(f"Processing {config['paths_file']}, found {len(path_list)} sets of images.")

        # if the results file already exists, do not process it again
        if not os.path.exists(config["results_file"]):
            for num, curr_set in tqdm(enumerate(path_list), total=len(path_list)):

                if curr_set.strip() != "" and not curr_set.startswith("#"):
                    curr_set_arr = curr_set.split(",")
                    # We load the images as numpy arrays
                    cell_data = []
                    if "r" in config["model_channels"]:
                        cell_data.append([imread(curr_set_arr[0].strip(), as_gray=True)])
                    if "y" in config["model_channels"]:
                        cell_data.append([imread(curr_set_arr[1].strip(), as_gray=True)])
                    if "b" in config["model_channels"]:
                        cell_data.append([imread(curr_set_arr[2].strip(), as_gray=True)])
                    if "g" in config["model_channels"]:
                        cell_data.append([imread(curr_set_arr[3].strip(), as_gray=True)])
                    # We run the model in inference
                    embedding, sig_probs, soft_probs = inference.run_model(model, cell_data, device)

                    if classifier_paths:
                        curr_probs_l = sig_probs.tolist()
                        max_location_class = curr_probs_l.index(max(curr_probs_l))
                        max_location_name = inference.CLASS2NAME[max_location_class]
                        max_3_location_classes = sorted(range(len(curr_probs_l)), key=lambda sub: curr_probs_l[sub])[
                            -3:]
                        max_3_location_classes.reverse()
                        max_3_location_names = (
                                inference.CLASS2NAME[max_3_location_classes[0]] + ","
                                + inference.CLASS2NAME[max_3_location_classes[1]] + ","
                                + inference.CLASS2NAME[max_3_location_classes[2]]
                        )

                    new_row = []
                    new_row.append(curr_set_arr[4].strip())
                    if classifier_paths:
                        new_row.append(max_location_name)
                        new_row.append(max_location_class)
                        new_row.append(max_3_location_names)
                        new_row.append(",".join(map(str, max_3_location_classes)))
                        new_row.extend(sig_probs)
                        new_row.extend(soft_probs)
                    new_row.extend(embedding)

                    df.loc[len(df.index)] = new_row

                    log_message = "- Saved results for " + curr_set_arr[4].strip()
                    if classifier_paths:
                        log_message = log_message + ", locations predicted [" + max_3_location_names + "]"
                    config["log"].info(log_message)

            df.to_csv(config['results_file'], index=False)
            print(f"Results saved to {config['results_file']}")
            self._process_results(df, config["log"], bg_masking, model_type, config['results_file'])

        config["log"].info("----------")
        config["log"].info("End: " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    def _process_results(self, df, log, bg_masking, model_type, results_file):

        parsed_df = df['id'].apply(parse_id).apply(pd.Series)
        df = pd.concat([df, parsed_df], axis=1)

        df['classification'] = df.apply(lambda x: 'Correct' if x['top_class_name'] == classification[x['protein']]
        else ('Acceptable' if acceptable_classification[x['protein']] is not None
                              and x['top_class_name'] in acceptable_classification[x['protein']]
              else "Wrong"),
                                        axis=1)

        log.info(f"Found {df['classification'].value_counts().to_dict()} classifications")
        log.info(f"Found {df['condition'].nunique()} conditions")
        log.info(f"Found {df['protein'].nunique()} proteins")

        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"Found NaNs in {file_name} for {prep}:")
            print(nan_counts[nan_counts > 0] / len(self.channels))
            # print the rows with NaNs
            nan_rows = df[df.isna().any(axis=1)]
            # remove all cols with prob or feat in them
            nan_rows = nan_rows.loc[:, ~nan_rows.columns.str.contains('prob|feat', case=False)]
            nan_rows = nan_rows.drop(columns=['top_class', 'top_3_classes', 'cell_id', 'protein', 'image_name'])
            print(nan_rows.head())
            raise ValueError(f"Found NaNs in {results_file} for {df['condition'].unique()}")

        # get the additional columns to the start of the dataframe
        additional_columns = ['condition', 'cell_id', 'protein', 'image_name', 'unique_cell_id', 'classification']
        # reorder the columns
        df = df[additional_columns + [col for col in df.columns if col not in additional_columns]]

        print(f"Processed file saved to {results_file}")
        df.to_csv(results_file, index=False, sep='\t')


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


def check_img_dims(img, patch_size):
    assert len(img.shape) == 4, f"Expected 4D tensor, got {len(img.shape)}D tensor"
    return einops.reduce(img, "c x y z -> c x y", "sum", z=1, x=patch_size.x, y=patch_size.y)


def preprocess_tensor(
    img,
    tile_spec,
    downsample_kernel=None,
    downsampling_method="mean",
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

    img = einops.rearrange(img, f"X Y Z C -> C Z Y X")

    # downsample kernel is 1 1 8 --> no downsampling on x and y, downsample on z
    if downsample_kernel is not None:
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

    # Create flat view (e.g. collaps zyx, but keep c which is 0 index of the tensor shape.
    img_flat = img.view(img.shape[0], -1)

    # Compute lower and upper percentile values for each channel. Notice that previously this was globally or the FOV
    # but here is it per cell as parts of the cell could be brighter than other parts. The assumption is that by doing
    # this, the model will focus more on structure and less on intensity (not tested).
    lower_values = torch.quantile(img_flat, 0.01, dim=1, keepdim=True)
    upper_values = torch.quantile(img_flat, 0.99, dim=1, keepdim=True)
    img_flat = torch.clamp(img_flat, min=lower_values, max=upper_values)

    # Avoid division by zero in case upper and lower values are equal
    range_values = upper_values - lower_values
    range_values = torch.where(range_values == 0, torch.ones_like(range_values), range_values)
    # Scale the clamped tensor to range [0, 1]
    img_flat = (img_flat - lower_values) / range_values

    # rearrange dims to C X Y Z
    img = einops.rearrange(img_flat, "C (Z Y X) -> C X Y Z", Z=tile_spec.z, Y=tile_spec.y, X=tile_spec.x)

    # check if all channels are 0 to 1 scaled
    for channel in img:
        assert channel.min() == 0 and (channel.max() == 1 or channel.max() == 0), f"Channel {channel} not in range [0, 1], got {channel.min()} to {channel.max()}"

    return img


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

