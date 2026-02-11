from loguru import logger


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
