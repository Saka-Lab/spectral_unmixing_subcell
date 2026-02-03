# Standard library imports
from pathlib import Path

# Third-party imports
import torch
# import einops
import numpy as np
import pandas as pd
from bioio import BioImage


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