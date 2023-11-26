#model_utils.py
# Helper functions for modeling

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance


def get_pipeline(FEATURE_TYPES, model_type):
    column_transformer = get_column_transformer(FEATURE_TYPES)
    normalized_column_transformer = get_column_transformer(FEATURE_TYPES, normalize=True)   
    if model_type == 'xgboost':
        xgbpipeln = Pipeline(steps = [
               ("column_transformer", column_transformer)
              ,("model",XGBClassifier(objective="binary:logistic", n_jobs=10, scale_pos_weight=9, use_label_encoder=False))
           ])
        return xgbpipeln

    if model_type == 'random_forest':
        rfpipeln = Pipeline(steps = [
               ("column_transformer", column_transformer)
              ,("model", RandomForestClassifier(n_jobs=10))
           ])
        return rfpipeln
    if model_type == 'svm':
        svm_pipeln = Pipeline(steps = [
               ("column_transformer", normalized_column_transformer)
              ,("model", SVC(class_weight="balanced", probability=True))
           ])
        return svm_pipeln
    if model_type == 'lgbm':
        LGBMClassifier_pipeln = Pipeline(steps = [
               ("column_transformer", column_transformer)
              ,("model", LGBMClassifier())
           ])
        return LGBMClassifier_pipeln


def get_hyperparams(model_type):
    if model_type == 'xgboost':
        xgbhyperparams = dict(model__subsample = [ 0.5], # 0.5 #final hyperparameters
                          model__reg_lambda = [ 0.01], # 0.01
                          model__n_estimators = [800], # 800
                          model__min_child_weight = [7], # 7
                          model__max_depth = [2],  # 2
                          model__learning_rate = [0.02], # 0.02
                          model__gamma = [2], # 2
                          model__colsample_bynode = [0.4] # 0.4
                          )
        return xgbhyperparams
    
    if model_type == 'random_forest':
        random_forrest_params = dict(
            model__n_estimators = [100,250],
            model__max_depth = [2,5,10],
            model__min_samples_leaf = [1,2,3],
            model__min_samples_split = [1,2,3,],
            #model__max_features = [0.5,0.6,0.7,0.8],
            model__class_weight = ["balanced"]
                )
        return random_forrest_params

    if model_type == 'svm':
        svm_params = dict(
                model__gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0],
                model__kernel = ["rbf"], 
                model__C = [0.1, 1, 10, 100, 1000], 
            )
        return svm_params

    if model_type == 'lgbm':
        LGBMClassifier_params = dict(
                model__n_estimators = [10,50,100,150,200,300],
                model__num_leaves = [10,50,75,100],
                model__max_depth = [1,2,3,4,5,6,7,8],
                model__learning_rate = [0.001,0.01,0.02,0.05,0.1,0.2]
            )
        return LGBMClassifier_params  






def get_final_features(FEATURE_TYPES, pipeline):

    final_features = list()

    if "interval" in FEATURE_TYPES.index:
        final_features += FEATURE_TYPES[["interval"]].tolist()

    if "count" in FEATURE_TYPES.index:
        final_features += FEATURE_TYPES[["count"]].tolist()
        
    # Build the list of nominal column names.
    if "nominal" in FEATURE_TYPES.index:
        nominal_categories = (
            pipeline.named_steps["column_transformer"]
            .named_transformers_["nominal_pipeline"]
            .named_steps["one_hot"]
            .categories_
        )
        nominal_categories = dict(
            zip(
                FEATURE_TYPES.loc[["nominal"]].tolist(),
                nominal_categories,
            )
        )
        nominal_columns = [
            "{}_{}".format(k, f) for (k, v) in nominal_categories.items() for f in v
        ]
        final_features += nominal_columns

    # Build the list of list column names.
    if "list" in FEATURE_TYPES.index:
        for list_feature in FEATURE_TYPES.loc[["list"]].tolist():
            columns = (
                pipeline.named_steps["column_transformer"]
                .named_transformers_["{0}_pipeline".format(list_feature)]
                .named_steps["encode"]
                .vocabulary_
            )
            columns = pd.Series(data=list(columns.keys()), index=list(columns.values()))
            columns = [
                "{0}_{1}".format(list_feature, v) for v in columns.sort_index().tolist()
            ]
            final_features += columns

    return final_features


def get_coefficients():
    raise RuntimeError("model_utils.get_coefficients() has not been implemented")

def get_importances(FEATURE_TYPES, pipeln, test_features,test_labels):
    feature_importances = pd.DataFrame(
        dict(
            Feature = get_final_features(FEATURE_TYPES, pipeln), 
            Importance = permutation_importance(pipeln, test_features, test_labels), 
            Importance_abs = np.abs(pipeln.named_steps["model"].feature_importances_))
    )
    
    feature_importances.sort_values("Importance_abs", ascending=False, inplace=True)
    feature_importances.reset_index(drop=True, inplace=True)
        
    return feature_importances



def get_feature_importances(FEATURE_TYPES, pipeln):
    feature_importances = pd.DataFrame(
        dict(
            Feature = get_final_features(FEATURE_TYPES, pipeln), 
            Importance = pipeln.named_steps["model"].feature_importances_, 
            Importance_abs = np.abs(pipeln.named_steps["model"].feature_importances_))
    )
    
    feature_importances.sort_values("Importance_abs", ascending=False, inplace=True)
    feature_importances.reset_index(drop=True, inplace=True)
        
    return feature_importances


def get_column_transformer(FEATURE_TYPES, normalize=False):

    if normalize == True:
        interval_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("normalize", StandardScaler()),
            ]
        )

        count_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("normalize", StandardScaler()),
            ]
        )
    else:
        interval_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])

        count_pipeline = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])

    nominal_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    list_pipeline = Pipeline(
        [
            (
                "encode",
                CountVectorizer(binary=False, lowercase=False, token_pattern=r"[^\s]+"),
            )
        ]
    )

    # NOTE: the order in which the feature types are processed MUST match 
    # the order they are processed in get_final_features()
    
    transformers = list()
    if "interval" in FEATURE_TYPES.index:
        transformers.append(
            (
                "interval_pipeline",
                interval_pipeline,
                FEATURE_TYPES.loc[["interval"]].tolist(),
            )
        )
    if "count" in FEATURE_TYPES.index:
        transformers.append(
            ("count_pipeline", count_pipeline, FEATURE_TYPES.loc[["count"]].tolist())
        )
    if "nominal" in FEATURE_TYPES.index:
        transformers.append(
            ("nominal_pipeline", nominal_pipeline, FEATURE_TYPES.loc[["nominal"]].tolist())
        )
    if "list" in FEATURE_TYPES.index:
        list_transformers = [
            ("{0}_pipeline".format(f), list_pipeline, f)
            for f in FEATURE_TYPES.loc[["list"]].tolist()
        ]
        transformers += list_transformers

    column_transformer = ColumnTransformer(transformers, n_jobs=10)

    return column_transformer


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
