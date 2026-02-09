# Standard library imports
from pathlib import Path

# Third-party imports
import torch
# import einops
import numpy as np
import pandas as pd
from loguru import logger
from bioio import BioImage


def get_experiments(input_dir):
    experiments = input_dir.iterdir()
    # filter out files, keep only directories
    experiments = [exp for exp in experiments if exp.is_dir()]
    # sort experiments by name
    experiments = sorted(experiments)
    if not experiments:
        raise ValueError(f"No experiments found in {input_dir}. Please check the directory structure.")
    logger.info(f"Found {len(experiments)} experiments in {input_dir}.")
    return experiments


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