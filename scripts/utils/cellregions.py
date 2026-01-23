# standard library imports
from pathlib import Path

# third-party imports
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm

# module imports
from utils.utils import Spec3D, load_img_data, get_regionprops, get_patch


class CellRegionDataset(Dataset):
    def __init__(
        self, img_dir, labels_dir, patch_size, bg_masking, h5_dir,
        transform=None, experiment_name=None, min_th: float = None, max_th: float = None,
    ):

        self.transform = transform
        self.patch_size = patch_size
        self.num_patches = 0
        self.bg_masking = bg_masking

        # storing precomputed cell regions
        self.h5f = None
        self.dataset_name = "cell_regions"

        exp_dir = img_dir / experiment_name
        lab_dir = labels_dir / experiment_name
        print(f"Loading images from {exp_dir} and labels from {lab_dir}...")

        self.hdf5_path = Path(h5_dir) / f"{experiment_name}_cell_regions.h5"
        img_paths = sorted(exp_dir.iterdir())
        
        # if there are manual segmentations use those
        # check if the manual segmentation directory exists
        lab_dir.mkdir(parents=True, exist_ok=True)
        # check if the hdf5 directory exists
        h5_dir.mkdir(parents=True, exist_ok=True)
        
        label_paths = sorted(lab_dir.iterdir())

        if len(label_paths) == 0:   # no manual segmentation
            print("No manual segmentations were found, fall back on processed masks")
            labels_dir = labels_dir.parent / labels_dir.name.replace("manual_", "")
            lab_dir = labels_dir / experiment_name
            label_paths = sorted(lab_dir.iterdir())


        img_paths = [im for im in img_paths if im.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')]
        label_paths = [im for im in label_paths if im.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')]

        print(f"Found {len(img_paths)} images and {len(label_paths)} labels")


        # check the correspondence between images and labels
        mismatch = False
        if len(img_paths) == len(label_paths):
            for i in range(len(img_paths)):
                if not img_paths[i].name == label_paths[i].name:
                    print("Mismatch between image and label names")
                    print(f"Image: {img_paths[i].name}\nLabel: {label_paths[i].name}")
                    mismatch = True
        else:
            mismatch = True
        if mismatch:
            print("Please make sure that the images and labels have the same names")
            print("Exiting...")
            raise ValueError("Mismatch between images and labels")

        imgs = [load_img_data(img_path, "CZYX") for img_path in img_paths]
        labels = [load_img_data(label_path, "ZYX") for label_path in label_paths]

        total_cells = 0
        all_regionprops = []
        for img, label, img_path in zip(imgs, labels, img_paths):
            if img.shape[1:] != label.shape:
                print(f"Something went wrong with the shapes skipping {img_path}")
                print(f"img shape: {img.shape[1:]}\tlabel shape: {label.shape}")
                continue
            original_shape = img.shape
            regionprops = get_regionprops(img, label, "CZYX", "ZYX")

            tqdm_total = (len(regionprops))
            total_cells += tqdm_total
            all_regionprops.append((regionprops, img_path.name, tqdm_total, original_shape))

        # save class attributes
        self.num_patches = total_cells

        # numer of sample in the dataset
        dataset_shape = (self.num_patches, *patch_size.tuple(), img.shape[0])    # nimages, patch, patch, z.stacks, channels
        print(f"Shape: {dataset_shape}")


        with h5py.File(self.hdf5_path, "w") as h5f:
            str_dtype = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset(self.dataset_name, dataset_shape, dtype=np.float32)
            h5f.create_dataset("image_name", (self.num_patches,), dtype=str_dtype)
            h5f.create_dataset("glob_coords", (self.num_patches, 4), dtype="float32")
            h5f.create_dataset("cell_idx_seg", (self.num_patches,), dtype=np.int32)
            idx = 0
            cell_idx = 0
            last_cell_idx = -1
            for regionprops, img_path, tqdm_total, original_shape in all_regionprops:
                print(f"Processing {img_path}...")
                img_name = '.'.join(img_path.split('.')[0:-1])
                for patch, local_cell_idx, centroid_coords, lab in tqdm(
                    self.img_generator(regionprops, patch_size, bg_masking), total=tqdm_total):

                    min_x = int(max(0, centroid_coords[0] - patch_size.x // 2))
                    min_y = int(max(0, centroid_coords[1] - patch_size.y // 2))
                    max_x = int(min(original_shape[3], centroid_coords[0] + patch_size.x // 2))
                    max_y = int(min(original_shape[2], centroid_coords[1] + patch_size.y // 2))
                    global_coords = (min_y, max_y, min_x, max_x)

                    h5f["cell_regions"][idx] = patch
                    h5f["image_name"][idx] = img_name
                    h5f["cell_idx_seg"][idx] = lab
                    h5f["glob_coords"][idx] = global_coords

                    # I expect multiple imgs returned for each cell
                    if last_cell_idx != local_cell_idx:
                        cell_idx += 1
                        last_cell_idx = local_cell_idx

                    idx += 1

        print(f"Saved {idx} samples to {self.hdf5_path}")

        
    def img_generator(self, regionprops, patch_size, bg_masking):
        for i, regionprop in enumerate(regionprops):
            # align patch to centroid
            patch_loc = Spec3D(
                int(regionprop.centroid_local[0]),
                int(regionprop.centroid_local[1]),
                int(regionprop.centroid_local[2]),
            )
            label = regionprop.label
 
            img = get_patch(regionprop, patch_loc, patch_size, bg_masking)
            yield img, i, regionprop.centroid, label

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_path, "r")
            self.data = self.h5f[self.dataset_name]

        img = self.data[idx]
        img = torch.from_numpy(img).float()

        img_name = self.h5f["image_name"][idx]
        label = self.h5f["cell_idx_seg"][idx]

        if self.transform is not None:
            img = self.transform(img)

        return img_name, label, img