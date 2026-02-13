import numpy as np
import tifffile
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from pathlib import Path

# Which PCC matrices should be computed
PCC_COMPARISONS = {
    "Gr1": ["Gr1", "Gr2", "Gr3"],
    "Gr2": ["Gr1", "Gr2", "Gr3"],
    "Gr3": ["Gr1", "Gr2", "Gr3"],
    "raw": ["Gr1", "Gr2", "Gr3"],
    "group": ["Gr1", "Gr2", "Gr3"],
    "full": ["Gr1", "Gr2", "Gr3"],
}

def extract_group_from_path(file_path: str) -> str:
    """
    Extract group name from a TIFF filename.
    Example: FOV1_Gr1.tif -> Gr1
             FOV1_raw.tif -> raw
    """
    stem = Path(file_path).stem  # remove .tif
    return stem.split("_")[-1]

def load_tif_image(file_path):
    """ Load a TIFF image stack and return it as a NumPy array. """
    with tifffile.TiffFile(file_path) as tif:
        # Load the entire stack (assuming Z-stack where each slice is a channel)
        image = tif.asarray()
    return image

def check_image_shape(image):
    """ Print the shape of the loaded image to debug the structure. """
    print(f"Loaded image shape: {image.shape}")
    return image.shape

def calculate_pcc_matrix(image1, image2):
    # Assuming image1 and image2 are NumPy arrays of shape (14, 993, 894)
    num_channels = image1.shape[0]
    pcc_matrix = np.zeros((num_channels, num_channels))
    
    for i in range(num_channels):
        for j in range(num_channels):
            channel1 = image1[i]
            channel2 = image2[j]

            # Convert images to float32 to avoid integer overflow during calculations
            channel1 = channel1.astype(np.float32)
            channel2 = channel2.astype(np.float32)

            # Apply gaussian filter sigma 1
            channel1 = gaussian_filter(channel1, 1.0)
            channel2 = gaussian_filter(channel2, 1.0)
            
            # Flatten the images to 1D arrays
            channel1_flat = channel1.flatten()
            channel2_flat = channel2.flatten()

            # Compute the Pearson correlation coefficient
            pearson_value, pearson_p = stats.pearsonr(channel1_flat, channel2_flat)
            pcc_matrix[i, j] = pearson_value
            
    return pcc_matrix

def create_pcc_dataframe(
    pcc_matrix,
    channel_names,
    group1,
    group2
):
    """
    Create a DataFrame from the PCC matrix with dynamic group labels.
    """
    df = pd.DataFrame(
        pcc_matrix,
        index=channel_names,
        columns=channel_names
    )

    df.index = [f"{name} - {group1}" for name in df.index]
    df.columns = [f"{name} - {group2}" for name in df.columns]

    return df

def pcc_from_tifs(
    image1_path: str,
    image2_path: str,
    channel_names: list[str],
    group1,
    group2
):
    # Load images
    image1 = load_tif_image(image1_path)
    image2 = load_tif_image(image2_path)

    # Extract group names from filenames
    group1 = extract_group_from_path(image1_path)
    group2 = extract_group_from_path(image2_path)

    # Check shapes
    shape1 = check_image_shape(image1)
    shape2 = check_image_shape(image2)

    if shape1 != shape2:
        raise ValueError("The shapes of the two images do not match!")

    num_channels = shape1[0]
    if num_channels != 14:
        raise ValueError(
            f"Expected 14 channels, but found {num_channels} channels."
        )

    # Calculate PCC matrix
    pcc_matrix = calculate_pcc_matrix(image1, image2)

    # Create DataFrame
    pcc_df = create_pcc_dataframe(
        pcc_matrix,
        channel_names,
        group1,
        group2
    )

    return pcc_df

def compute_all_pcc_for_single_fov(
    fov_dir,
    channel_names,
    output_subdir="pcc",
    comparison_rules=PCC_COMPARISONS
):
    fov_dir = Path(fov_dir)
    fov_name = fov_dir.name

    output_dir = fov_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect TIFFs
    tif_paths = list(fov_dir.glob("*.tif"))
    if not tif_paths:
        raise ValueError(f"No TIFF files found in {fov_dir}")

    group_to_path = {
        extract_group_from_path(p): p
        for p in tif_paths
    }

    # Validate that required groups exist
    for g1, g2_list in comparison_rules.items():
        if g1 not in group_to_path:
            raise ValueError(f"Missing TIFF for group '{g1}'")
        for g2 in g2_list:
            if g2 not in group_to_path:
                raise ValueError(f"Missing TIFF for group '{g2}'")

    # Compute requested PCCs
    for group1, group2_list in comparison_rules.items():
        for group2 in group2_list:

            print(f"Computing PCC: {group1} vs {group2}")

            pcc_df = pcc_from_tifs(
                image1_path=group_to_path[group1],
                image2_path=group_to_path[group2],
                channel_names=channel_names,
                group1=group1,
                group2=group2
            )

            output_name = (
                f"{fov_name}_{group1}_vs_{group2}_gb_ncc_matrix.csv"
            )
            output_path = output_dir / output_name
            pcc_df.to_csv(output_path)