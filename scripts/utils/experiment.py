from bioio import BioImage
import numpy as np
from bioio.writers import OmeTiffWriter
import os
from pathlib import Path

# micro_sam imports
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

# mask processing
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage.morphology import erosion, disk, dilation

from utils.cellregions import CellRegionDataset

from utils.utils import check_img_dims, protein_image_generator, repack_h5_file
import torchvision
import torch
import pandas as pd
import h5py

import logging
import datetime
from utils.utils import get_model
from skimage.io import imread
import utils.inference as inference
from utils.utils import parse_id

# correct classes:

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


class Preprocess:
    def __init__(
        self, name, exp_dir, channels, 
        min_th, max_th,
        seg_channels, seg_model, n_dim,
        min_cell_area, min_border_area,
        tmp_dir='tmp', res_dir='preprocessing_results'
        ):
        self.name = name
        self.exp_dir = exp_dir
        self.channels = channels

        # preprocessing
        self.min_th = min_th
        self.max_th = max_th

        # segmentation
        self.seg_channels = seg_channels
        self.seg_model = seg_model
        self.n_dim = n_dim

        # mask processing
        self.min_cell_area = min_cell_area
        self.min_border_area = min_border_area

        # folders setup
        self.tmp_dir = tmp_dir
        self.res_dir = res_dir

    def __repr__(self):
        return f"""Experiment(
        name={self.name}, exp_dir={self.exp_dir},
        channels={self.channels},
        min_th={self.min_th}, max_th={self.max_th},
        seg_channels={self.seg_channels}, seg_model={self.seg_model}, n_dim={self.n_dim},
        min_cell_area={self.min_cell_area}, min_border_area={self.min_border_area},
        tmp_dir={self.tmp_dir}, res_dir={self.res_dir}
        )\n"""
        

    def _save_image(self, image_data, save_path, channel_names=None):
        # Save the processed image data to the specified path
        print(f"Saving image to: {save_path}")
        # check if the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(image_data.shape) == 4:
            dim_order = "CZYX"
        elif len(image_data.shape) == 3:
            dim_order = "ZYX"
        else:
            raise ValueError(f"Unsupported image shape: {image_data.shape}")
        OmeTiffWriter.save(
            image_data,
            save_path,
            dim_order=dim_order,  # must match your array shape
            channel_names=channel_names,
            image_name=save_path.stem,
        )

    # ---------- Image preprocessing ----------
    def _clamp_and_convert(self, img_path, channel_names):
        clamp_path = self.res_dir/"8bits"/f"{self.min_th}_{self.max_th}"/self.name/img_path.name
        # if the file already exists skip
        if clamp_path.exists():
            return clamp_path

        img_data = BioImage(img_path).data
        # BioImage is a 5D array: T C Z Y X
        img_data = img_data[0]

        normalized_channels = []
        C = img_data.shape[0]
        for c in range(C):
            channel = img_data[c]
            # find the value at the th percentile
            max_value_th = int(np.percentile(channel, float(self.max_th)))
            min_value_th = int(np.percentile(channel, float(self.min_th)))
            min_value_th = max(1, min_value_th)  # ensure min_th is at least 1 to at least remove something
            channel[channel > max_value_th] = max_value_th
            channel[channel <= min_value_th] = min_value_th
            ch_min, ch_max = np.min(channel), np.max(channel)
            if ch_min != ch_max:
                channel = (((channel - ch_min) / (ch_max - ch_min)) * 255).astype(np.uint8)
            normalized_channels.append(channel)

        image_data = np.stack(normalized_channels, axis=0)
        self._save_image(image_data, clamp_path, channel_names)
        return clamp_path


    def _convert_to_greyscale(self, img_path):
        greyscale_path = self.res_dir/"greyscale"/f"{self.min_th}_{self.max_th}"/self.name/img_path.name

        if greyscale_path.exists():
            return greyscale_path

        # Convert the image data to greyscale using the specified segmentation channels
        img_data = BioImage(img_path).data
        # BioImage is a 5D array: T C Z Y X
        img_data = img_data[0]


        # find the indexes of the channels to use for segmentation
        idxs = [self.channels.index(channel) for channel in self.seg_channels]
        # only keep the channels to use for segmentation
        img_data = img_data[idxs]
        img_data = np.sum(img_data, axis=0)
        min_val, max_val = np.min(img_data), np.max(img_data)
        img_data = (img_data - min_val) / (max_val - min_val) * 255

        self._save_image(img_data, greyscale_path)

        return greyscale_path

    def _segment_image(self, img_path):
        seg_path = self.tmp_dir/"segmentations"/self.name/img_path.name

        if seg_path.exists():
            return seg_path
        if not seg_path.parent.exists():
            seg_path.parent.mkdir(parents=True)

        predictor, segmentor = get_predictor_and_segmenter(model_type=self.seg_model)

        instances = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmentor,
                input_path=img_path,
                output_path=seg_path,
                ndim=self.n_dim,
            )
        return seg_path

    def _clean_mask(self, mask_path):
        cleaned_mask_path = self.res_dir/"segmentations"/self.name/mask_path.name

        if cleaned_mask_path.exists():
            return cleaned_mask_path
        if not cleaned_mask_path.parent.exists():
            cleaned_mask_path.parent.mkdir(parents=True)

        disk_size = 4
        disk_shape = disk(disk_size)

        mask_data = BioImage(mask_path).data
        mask_data = mask_data[0][0]
        try:
            Z, H, W = mask_data.shape
        except ValueError:
            print("Unexpected mask shape:", mask_data.shape)
            if len(mask_data.shape) == 4 and mask_data.shape[0] == 1:
                mask_data = mask_data[0]
                Z, H, W = mask_data.shape
            else:
                raise ValueError(f"Unsupported mask shape: {mask_data.shape}")
        areas = []
        unique_labels = [label for label in np.unique(mask_data) if label != 0] # skip the background label
        labels_to_keep = []
        # Analyze slice z=0 for filtering
        base_slice = mask_data[0]
        for label in unique_labels:
            slice_mask = base_slice == label
            area = np.sum(slice_mask)
            coords = np.argwhere(slice_mask)
            touches_border = np.any(
                (coords[:, 0] == 0) | (coords[:, 0] == H - 1) |
                (coords[:, 1] == 0) | (coords[:, 1] == W - 1)
            )

            if area >= self.min_cell_area and not touches_border:
                labels_to_keep.append(label)

        output_mask = np.zeros_like(mask_data)
        for label in tqdm(labels_to_keep):
            # Vectorized 3D processing
            label_mask_3d = mask_data == label
            for z in tqdm(range(Z), leave=False):
                slice_mask = binary_fill_holes(label_mask_3d[z])
                slice_mask = erosion(slice_mask, disk_shape)
                slice_mask = dilation(slice_mask, disk_shape)
                output_mask[z][slice_mask] = label

        self._save_image(output_mask, cleaned_mask_path)
        return cleaned_mask_path

    def process_images(self):
        """
        Process images for the experiment.
        This method will separate the images into scenes and save them in the temporary directory.
        """
        print(f"\n=== Running {self.name} ===")
        image_files = sorted(self.exp_dir.glob("*.tif"))
 
        if not image_files:
            print(f"No images found for {self.name} in {self.exp_dir}")
            return
        else:
            print(f"Found {len(image_files)} images.")

        for img_path in image_files:
            print(f"Processing image: {img_path}")
            clamped = self._clamp_and_convert(img_path, self.channels)

            # if manual segmentations already exist skip greyscale, segmentation and mask processing
            if (self.res_dir/"manual_segmentations"/self.name/img_path.name).exists():
                print(f"Manual segmentation already exists: {(self.res_dir/'manual_segmentations'/self.name/img_path.name)}")
                continue
            if (self.res_dir/"segmentations"/self.name/img_path.name).exists():
                 print(f"Segmentation image already exists: {(self.res_dir/'segmentations'/self.name/img_path.name)}")
                 continue
            greyscale = self._convert_to_greyscale(clamped)
            mask = self._segment_image(greyscale)
            cleaned_mask = self._clean_mask(mask)


class Subcell:

    def __init__(self, name, min_th, max_th, transform, cell_patch_size, channels, ref_channels, model_channels, input_dir, prep_dir, res_dir, gpu):
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
            min_th=self.min_th,
            max_th=self.max_th,
        )


        protein_channel_map = {channel: i for i, channel in enumerate(self.channels)}
        ref_channel_map = {channel: self.channels.index(channel) if channel in self.channels else "" for channel in self.ref_channels}
        ref_channel_keys = [key for key, channel in ref_channel_map.items() if channel != ""]
        selected_channel_map = {key: ref_channel_map[key] for key in ref_channel_map if ref_channel_map[key] != ""}

        img_data = []
        img_path = Path(out_path)/"imgs"
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
                    p = Path(img_path)/f"{img_name.decode()}_{str(lab)}_{p_names[i]}.png"
                    img = img.unsqueeze(0)
                    torchvision.io.write_png((img * 255).to(torch.uint8), p, compression_level=9)
                    channel_imgs.append(p)

                img_data.append({
                        "r_image": channel_imgs[ref_channel_keys.index("alphaTUBULIN")] if "alphaTUBULIN" in ref_channel_keys else "",
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
        paths_file = Path(self.prep_dir)/f"{self.min_th}_{self.max_th}"/self.name/f"path_list_{self.name}.csv"
        results_file = Path(self.res_dir)/f"{self.min_th}_{self.max_th}"/f"{self.name}{'_bgmask' if bg_masking else ''}_{model_type.split('_')[0]}.tsv"

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
            file_handler = logging.FileHandler(f"logs/{self.name}_{model_type}{'_bg_masking' if bg_masking else ''}_log.txt", encoding="utf-8", mode="w")
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
                        max_3_location_classes = sorted(range(len(curr_probs_l)), key=lambda sub: curr_probs_l[sub])[-3:]
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
            print(nan_counts[nan_counts > 0]/len(self.channels))
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