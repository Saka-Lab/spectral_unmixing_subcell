# Subcellular Classification Output

## Overview

This dataset contains the results of subcellular localization classification for individual cells. Each row represents a single cell analyzed for protein localization patterns, with associated metadata, classification results, and feature embeddings.

## File Format

- **Format**: CSV (Comma-Separated Values)
- **Structure**: One row per cell
- **Columns**: 79 columns total

## Column Descriptions

### Metadata Columns

| Column | Description                                                                           |
|--------|---------------------------------------------------------------------------------------|
| `condition` | Experimental condition (e.g., ActD for Actinomycin D treatment)                       |
| `cell_id` | Numerical identifier for the cell within the image. Corresponds to segmentation label |
| `protein` | Protein marker or channel analyzed (e.g., DAPI for nuclear staining)                  |
| `image_name` | Full filename of the source image                                                     |
| `unique_cell_id` | Globally unique identifier combining image name and cell ID                           |

### Classification Results

| Column | Description |
|--------|-------------|
| `classification` | Ground truth or validation label (e.g., "Correct", "Incorrect") |
| `id` | Full unique identifier for this specific classification instance |
| `top_class_name` | Human-readable name of the predicted subcellular location (e.g., "Nucleoplasm", "Nuclear membrane") |
| `top_class` | Numerical class index for the top prediction |
| `top_3_classes_names` | Comma-separated names of the top 3 predicted classes |
| `top_3_classes` | Comma-separated indices of the top 3 predicted classes |

#### Sigmoid Probabilities (`sig_prob00` - `sig_prob30`)
- Independent probability for each class (probabilities don't sum to 1)
- Useful for multi-label scenarios where a protein can localize to multiple compartments
- Values range from 0 to 1

#### Softmax Probabilities (`soft_prob00` - `soft_prob30`)
- Mutually exclusive probabilities (sum to 1 across all classes)
- Standard classification probabilities
- Values range from 0 to 1
- The highest value corresponds to `top_class`

### Feature Embeddings (`feat0000` - `feat1535`)

Low-dimensional feature representations extracted from the classification model:
- These are typically the output of the last layer of the neural network
- Can be used for clustering, visualization (UMAP/t-SNE), or downstream analysis
- Number of features may vary by model. Number of features here is with the default workflow on data used in the paper.
- Values are continuous and can be positive or negative

## Example Record

```csv
condition: ActD
cell_id: 5
protein: DAPI
top_class_name: Nucleoplasm
top_class: 26
top_3_classes_names: Nucleoplasm,Nuclear membrane,Nuclear bodies
```

This example shows a DAPI-stained cell under Actinomycin D treatment, classified as localizing primarily to the Nucleoplasm (class 26), with Nuclear membrane and Nuclear bodies as alternative possibilities.

## Subcellular Localization Classes

The model predicts across 31 different subcellular compartments (indexed 0-30). Common classes include:

- Nucleoplasm
- Nuclear membrane
- Nuclear bodies
- Cytoplasm
- Plasma membrane
- Mitochondria
- Endoplasmic reticulum
- Golgi apparatus
- (and 23+ other compartments)

*Note: The exact mapping of class indices to localization names depends on your specific model training.*

## Usage Notes

1. **Multi-label analysis**: Use `sig_prob` columns when investigating proteins that may localize to multiple compartments simultaneously.

2. **Single-label classification**: Use `soft_prob` columns for standard classification tasks where each cell is assigned to exactly one compartment.

3. **Uncertainty quantification**: Compare `top_class` probability with the second and third highest probabilities to assess prediction confidence.

4. **Dimensionality reduction**: The `feat` columns are ideal for UMAP or t-SNE visualization to explore the relationship between different cell populations.

5. **Quality control**: The `classification` column can be used to filter or validate predictions against ground truth labels.

## Classification Certainty Indicators

- **High confidence**: When the top softmax probability is >0.7
- **Ambiguous**: When top 2-3 classes have similar probabilities
- **Multi-localization**: When multiple sigmoid probabilities are >0.5