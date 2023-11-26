

#project_utils.py
#Project helper functions

import logging
import os
import sys
from uuid import uuid4

def set_project_dir(project):    
    if os.getcwd().find(project) == -1:
        raise RuntimeError(f"The target project directory '{project}' was not found in the path of the current working directory.")
    else:
        project_path = os.getcwd().split(project, 1)[0] + project 
    os.chdir(project_path)

    return project_path


def get_model_dirs(output_root_path, model_id):
    output_path = f"{output_root_path}\\{model_id}"
    training_data = f"{output_path}\\training_data"
    training_figures = f"{output_path}\\training_figures"
    model_file = f"{output_path}\\model_{model_id}.dat"
    future_predictions = f"{output_path}\\future_predictions"

    dirs = {
        "output": output_path,
        "log": output_path,
        "training_data": training_data,
        "training_figures": training_figures,
        "model_file": model_file,
        "future_predictions": future_predictions
        }

    return dirs


def create_new_model_dirs(output_root_path):
    model_id = str(uuid4()) # Generate new GUID each time a model fit is performed

    dirs = get_model_dirs(output_root_path, model_id)
    os.mkdir(dirs["output"])
    os.mkdir(dirs["training_data"])
    os.mkdir(dirs["training_figures"])
    os.mkdir(dirs["future_predictions"])

    print("Output directories created:")
    print(f"    {dirs['output']}")
    print(f"    {dirs['training_data']}")
    print(f"    {dirs['training_figures']}")
    print(f"    {dirs['future_predictions']}")

    return model_id, dirs


def get_future_predictions_dir(output_root_path, model_id, subdir_prefix, future_predictions_id):
    dirs = get_model_dirs(output_root_path, model_id)
    dirs["future_predictions_subdir"] = f"{dirs['future_predictions']}\\{subdir_prefix}___{future_predictions_id}"

    return dirs


def create_new_future_predictions_dir(output_root_path, model_id, subdir_prefix):
    future_predictions_id = str(uuid4()) # Generate new GUID each time future predictions are made

    dirs = get_future_predictions_dir(
        output_root_path=output_root_path, 
        model_id=model_id, 
        subdir_prefix=subdir_prefix, 
        future_predictions_id=future_predictions_id
        )
    os.mkdir(dirs["future_predictions_subdir"])

    print("Output directory created:")
    print(f"    {dirs['future_predictions_subdir']}")

    return future_predictions_id, dirs


def get_basic_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir + "\\log.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger
