import os
import yaml
from box.exceptions import BoxValueError
from src.logger.custom_logging import logger
from ensure import ensure_annotations
from pathlib import Path
from box import ConfigBox
import pickle
import json
from src.exceptions.expection import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import sys

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb')as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)


# def evaluate_model(models, X_train, X_test, y_train, y_test, params):
#     try:
#         report = {}

#         # Loop through models and evaluate them
#         for model_name, model in models.items():
#             # Fetch corresponding parameters from the dictionary
#             param = params.get(model_name, {})

#             logger.info(f"Starting RandomizedSearchCV for {model_name}")
#             # Initialize RandomizedSearchCV
#             rs = RandomizedSearchCV(model, param, cv=7, random_state=42, n_jobs=-1, n_iter=100)

#             try:
#                 # Fit the model
#                 rs.fit(X_train, y_train)
#                 logger.info(f"Best parameters for {model_name}: {rs.best_params_}")

#                 # Set the best parameters for the model and fit again
#                 model.set_params(**rs.best_params_)
#                 model.fit(X_train, y_train)

#                 # Predict and evaluate the model
#                 y_pred = model.predict(X_test)
#                 test_model_score = r2_score(y_test, y_pred)

#                 # Store the results
#                 report[model_name] = test_model_score

#             except Exception as inner_e:
#                 logger.error(f"Error occurred while tuning {model_name}: {inner_e}")
#                 report[model_name] = None  # Mark as None in case of error during tuning

#         return report

#     except Exception as e:
#         logger.error(f"Error occurred: {e}")
#         raise CustomException(e, sys) 
    


# def final_model(models, X_train, X_test, y_train, y_test, params_path='params.yaml'):
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)

#         report = {}
#         for model_name, model in models.items():
#             param = params.get(model_name, {})

#             logger.info(f"Starting RandomizedSearchCV for {model_name}")
#             rs = RandomizedSearchCV(model, param, cv=7, random_state=42, n_jobs=-1, n_iter=100)

#             try:
#                 rs.fit(X_train, y_train)
#                 logger.info(f"Best parameters for {model_name}: {rs.best_params_}")

#                 model.set_params(**rs.best_params_)
#                 model.fit(X_train, y_train)

#                 y_pred = model.predict(X_test)
#                 test_model_score = r2_score(y_test, y_pred)

#                 report[model_name] = test_model_score

#             except Exception as inner_e:
#                 logger.error(f"Error occurred while tuning {model_name}: {inner_e}")
#                 report[model_name] = None

#         return report

#     except Exception as e:
#         logger.error(f"Error occurred: {e}")
#         raise CustomException(e, sys)