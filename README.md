This repository provides an end-to-end workflow to go from raw microscopy images to SubCell classification results and interactive UMAP visualizations.

Raw images (LIF / TIFF)
        ↓
1. split_lif
        ↓
2. preprocessing
        ↓
3. subcell
        ↓
4. hv_plot
        ↓
Interactive dashboard (Panel) and/or SVG figures


- split_lif.py, split raw image files containing multiple scenes.

- preprocessing, prepares to run subcell with preprocessing, segmenting, and mask post-processing.

- subcell, prepares the dataset for subcell, and runs it.

- hv_plot, creates an interactive plot and saves an svg file.