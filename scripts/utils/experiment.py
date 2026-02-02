from bioio import BioImage
import numpy as np
from bioio.writers import OmeTiffWriter

# micro_sam imports
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

# mask processing
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage.morphology import erosion, disk, dilation


from utils.utils import parse_id
from loguru import logger


# correct classes:


class Preprocess:
    def __init__(
        self, name, exp_dir, channels, 
        min_th, max_th,
        seg_channels, seg_model, n_dim,
        min_cell_area,
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
        logger.info(f"Saving image to: {save_path}")
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

        img_data = BioImage(img_path).get_image_data("CZYX")

        normalized_channels = []
        num_ch = img_data.shape[0]
        for ch_ind in range(num_ch):
            channel = img_data[ch_ind]
            # find the value at the th percentile
            max_value_th = int(np.percentile(channel, self.max_th))
            min_value_th = int(np.percentile(channel, self.min_th))
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

        automatic_instance_segmentation(
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
            logger.debug("Unexpected mask shape:", mask_data.shape)
            if len(mask_data.shape) == 4 and mask_data.shape[0] == 1:
                mask_data = mask_data[0]
                Z, H, W = mask_data.shape
            else:
                raise ValueError(f"Unsupported mask shape: {mask_data.shape}")

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
        logger.info(f"\n=== Running {self.name} ===")
        image_files = sorted(self.exp_dir.glob("*.tif"))
 
        if not image_files:
            logger.debug(f"No images found for {self.name} in {self.exp_dir}")
            return
        else:
            logger.info(f"Found {len(image_files)} images.")

        for img_path in image_files:
            logger.info(f"Processing image: {img_path}")
            clamped = self._clamp_and_convert(img_path, self.channels)

            # if manual segmentations already exist skip greyscale, segmentation and mask processing
            if (self.res_dir/"manual_segmentations"/self.name/img_path.name).exists():
                logger.info(f"Manual segmentation already exists: {(self.res_dir/'manual_segmentations'/self.name/img_path.name)}")
                continue
            if (self.res_dir/"segmentations"/self.name/img_path.name).exists():
                 logger.info(f"Segmentation image already exists: {(self.res_dir/'segmentations'/self.name/img_path.name)}")
                 continue
            greyscale = self._convert_to_greyscale(clamped)
            mask = self._segment_image(greyscale)
            self._clean_mask(mask)
