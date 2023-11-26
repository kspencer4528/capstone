# Train and evaluate the model

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline


# Import Internal Modules
from shpretention.model_utils import get_column_transformer, get_pipeline, get_hyperparams
from shpretention.plot_utils import (plot_best_threshold_confusion_matrix,
                                     plot_feature_importance,
                                     plot_prediction_distributions,
                                     plot_roc_curves, 
                                     plot_pr_curves, 
                                     plot_pr_threshold_curves, 
                                     plot_confusion_matrix,
                                     get_precision_recall,
                                     get_threshold)
from shpretention.project_utils import (set_project_dir, 
                                        create_new_model_dirs, 
                                        get_basic_logger)
from shpretention.data_utils import get_features

#%% Configure project details

set_project_dir(project="shp_retention")
model_type = "xgboost" # xgboost, svm, random_forest, lgbm
output_root = os.path.join("C:/repos/shp_retention/output/" + model_type)
dsn = "AWSDW"
num_years = 2
window_start = "2020-01-01"
window_stop = "2020-08-31"

#%% Set up directories and create logger
# Get new model ID and create output folders
model_id, dirs = create_new_model_dirs(
    output_root
)

# Create logger that saves to output folder
logger = get_basic_logger(dirs["log"])

logger.info("""Started running train_and_evaluate.py""")
logger.info(f"""   output_root: {output_root}""")
logger.info(f"""   dsn: {dsn}""")
logger.info(f"""   model_id: {model_id} """)

#%% Specify features and label
FEATURES = pd.DataFrame(
    data = np.array(
        [
            # Detail only
            ["shp_member_id", ""],
            # Numeric count
            ["contact_count", "count"],
            ["paid_count", "count"],
            ["denied_count", "count"],
            ["cc_disease_cnt", "count"],
            # Numeric continuous
            ["age", "interval"],
            ["rpl_theme1", "interval"],
            ["rpl_theme2", "interval"],
            ["rpl_theme3", "interval"],
            ["rpl_theme4", "interval"],
            ["rpl_themes", "interval"],
            ["denied_amt", "interval"],
            ["allowed_amt", "interval"],
            ["member_amt", "interval"],
            ["distance_in_km", "interval"],
            # Categorical
            ["gender_code", "nominal"],
            ["member_physical_zip_code", "nominal"],
            ["county_name", "nominal"],
            ["provider_organization_name", "nominal"],
            ["practitioner_full_name", "nominal"],
            ["practitioner_primary_region", "nominal"],
            ["practitioner_provider_type_code", "nominal"],
            ["product_type_code", "nominal"],
            ["product_name", "nominal"],
            ["member_esrd_flag", "nominal"],
            ["part_c_flag", "nominal"],
            ["part_d_flag", "nominal"],
            ["cchg_cat_rollup_code", "nominal"],
            ["cchg_cat", "nominal"],
            ["cancer_severity_code", "nominal"],
            ["copd_severity_code", "nominal"],
            ["diabetes_severity_code", "nominal"],
            ["hypertension_severity_code", "nominal"],
            ["high_opioid_usage_flag", "nominal"],
            ["polypharmacy_status_flag", "nominal"],
            ["frequent_er_flag", "nominal"],
            ["frequent_imaging_flag", "nominal"],
            ["frequent_inpatient_admission_flag", "nominal"],
            ["comorbidity_flag", "nominal"],
            

        ]
    ),
    columns=["name", "type"],
)

FEATURE_TYPES = FEATURES.set_index("type")["name"]

LABEL = "quit_plan"

#%% Get features and labels

cat_features = FEATURE_TYPES.loc[["nominal"]].tolist() 

traintest_features = pd.DataFrame()
for year in range(0, num_years):
    ma_data = get_features(
        logger=logger, 
        dsn=dsn, 
        window_start=window_start, 
        window_stop=window_stop,
        cat_features=cat_features
    )    
    traintest_features = pd.concat([traintest_features, ma_data])
    window_start = (pd.to_datetime(window_start)+pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    window_stop = (pd.to_datetime(window_stop)+pd.DateOffset(years=1)).strftime("%Y-%m-%d")

traintest_features = traintest_features.drop_duplicates(subset=["shp_member_id"], keep="last")

# Create a surrogate key for the features
traintest_features.insert(
    0,
    "features_id",
    traintest_features.index.to_series().map(lambda _: str(uuid4())),
)
traintest_features = traintest_features.reset_index(drop=True)
traintest_labels = traintest_features[["features_id", LABEL]]
traintest_features = traintest_features.drop(LABEL, axis=1)

traintest_features = traintest_features.set_index("features_id")
traintest_labels = traintest_labels.set_index("features_id")

# Split the data into train and test
train_features, test_features, train_labels, test_labels = train_test_split(
    traintest_features, traintest_labels, test_size=0.2, stratify=traintest_labels[LABEL], random_state=None
)

logger.info("DataFrames created:")
logger.info(f"    traintest_features with shape of {traintest_features.shape}.")
logger.info(f"    traintest_labels with shape of {traintest_labels.shape}.")
logger.info(f"    train_features with shape of {train_features.shape}.")
logger.info(f"    train_labels with shape of {train_labels.shape}.")
logger.info(f"    test_features with shape of {test_features.shape}.")
logger.info(f"    test_labels with shape of {test_labels.shape}.")
logger.info(f"    Percentage qutit plan:")
logger.info(f"    Train data: {round(100*train_labels[LABEL].values.sum()/train_labels[LABEL].count(), 2)}% have {LABEL}.")
logger.info(f"    Test data: {round(100*test_labels[LABEL].values.sum()/test_labels[LABEL].count(), 2)}% have {LABEL}.")

#%% Create model

pipeln=get_pipeline(FEATURE_TYPES,model_type)

#%% Perform grid search
hyper_params=get_hyperparams(model_type)

logger.info("Training model with grid search...")

start_time = datetime.now()

n_iter=1000
cv=3
n_jobs=10
metric="roc_auc"

grid = RandomizedSearchCV(
            estimator=pipeln,
            param_distributions=hyper_params,
            scoring=["roc_auc", "average_precision"],
            cv=cv,
            n_jobs=n_jobs,
            n_iter=n_iter,
            return_train_score=True,
            refit=metric,
        )

logger.info(f"    n_iter: {n_iter}")
logger.info(f"    cv: {cv}")
logger.info(f"    n_jobs: {n_jobs}")
logger.info(f"    metric: {metric}")

search = grid.fit(train_features,
                  train_labels[LABEL])

# If the current model already exists, get new model_id and create new output directories
if "model_filename" in locals() and os.path.isfile(Path(dirs["output"], eval("model_filename"))):
    old_model_id = model_id
    model_id, dirs = create_new_model_dirs(output_root)
    logger.info(f"Switching from existing model {old_model_id} to new model {model_id}...")
    logger = get_basic_logger(dirs["log"])
    logger.info(f"Switched from existing model {old_model_id} to new model {model_id}...")

pipeln = grid.best_estimator_

runtime = datetime.now() - start_time

logger.info(f"...finished in {round(runtime.total_seconds(),1)} seconds!")
logger.info(f"Best hyperparameters:\n{json.dumps(search.best_params_, indent=4)}")

#%% Get predictions
train_preds = pd.DataFrame(
    data={
        "features_id": train_features.reset_index()["features_id"],
        "prediction": pipeln.predict_proba(train_features)[:, 1]
    }
).set_index("features_id")

test_preds = pd.DataFrame(
    data={
        "features_id": test_features.reset_index()["features_id"],
        "prediction": pipeln.predict_proba(test_features)[:, 1]
    }
).set_index("features_id")

logger.info("Prediction quantiles:")
logger.info( "    Data:   Train    Test")
logger.info(f"    Min:    {float(train_preds.min()):.4f}   {float(test_preds.min()):.4f}")
logger.info(f"    Q1:     {float(train_preds.quantile(0.25)):.4f}   {float(test_preds.quantile(0.25)):.4f}")
logger.info(f"    Median: {float(train_preds.quantile(0.50)):.4f}   {float(test_preds.quantile(0.50)):.4f}")
logger.info(f"    Q3:     {float(train_preds.quantile(0.75)):.4f}   {float(test_preds.quantile(0.75)):.4f}")
logger.info(f"    Max:    {float(train_preds.max()):.4f}   {float(test_preds.max()):.4f}")

#%% Save model, features, labels, and predictions

# Save model
pickle.dump(pipeln, open(Path(dirs["model_file"]), "wb"))

# Save features
FEATURES.to_csv(Path(dirs["training_data"], "FEATURES.csv"), index=False)
train_features[FEATURES.name].to_csv(Path(dirs["training_data"], "train_features.csv"), index=True)
test_features[FEATURES.name].to_csv(Path(dirs["training_data"], "test_features.csv"), index=True)

# Save unused features
if np.any(~train_features.columns.isin(FEATURES.name)):
    train_features.iloc[:,~train_features.columns.isin(FEATURES.name)].to_csv(Path(dirs["training_data"], "unused_train_features.csv"), index=True)
    test_features.iloc[:,~test_features.columns.isin(FEATURES.name)].to_csv(Path(dirs["training_data"], "unused_test_features.csv"), index=True)

# Save labels
train_labels.to_csv(Path(dirs["training_data"], "train_labels.csv"), index=True)
test_labels.to_csv(Path(dirs["training_data"], "test_labels.csv"), index=True)

# Save predictions
train_preds.to_csv(Path(dirs["training_data"], "train_preds.csv"), index=True)
test_preds.to_csv(Path(dirs["training_data"], "test_preds.csv"), index=True)
logger.info("Model, features, labels, and predictions saved to output folder.")

#%% Create and save figures 
# Prediction distributions split by label
fig = plot_prediction_distributions(train_labels, train_preds, test_labels, test_preds, hue="label")
fig_path = Path(dirs["training_figures"], "prediction_distributions.png")
fig.savefig(fig_path, bbox_inches="tight")

# Prediction distributions not split by label
fig = plot_prediction_distributions(train_labels, train_preds, test_labels, test_preds, hue=None)
fig_path = Path(dirs["training_figures"], "prediction_distributions_without_label_hue.png")
fig.savefig(fig_path, bbox_inches="tight")

# Feature importances
if model_type != 'svm':
    fig = plot_feature_importance(FEATURE_TYPES, pipeln)
    fig_path = Path(dirs["training_figures"], "feature_importance.png")
    fig.savefig(fig_path, bbox_inches="tight")
 else:
     fig = plot_perm_importances(FEATURE_TYPES, pipeln,test_features, test_labels)
     fig_path = Path(dirs["training_figures"], "feature_importance.png")
     fig.savefig(fig_path, bbox_inches="tight")    

# Train/test ROC AUCs
fig, train_roc_auc, test_roc_auc = plot_roc_curves(train_labels, train_preds, test_labels, test_preds)
fig_path = Path(dirs["training_figures"], "roc_curves.png")
fig.savefig(fig_path, bbox_inches="tight")

logger.info("ROC AUC scores:")
logger.info(f"    Train AUC: {train_roc_auc:.4f}.")
logger.info(f"    Test AUC: {test_roc_auc:.4f}.")

precision_recall = get_precision_recall(test_labels, test_preds)
logger.info(f"Precision Recall : ")
logger.info(f"                  {precision_recall}")

# Confusion matrices
fig = plot_confusion_matrix(test_labels, test_preds)
fig_path = Path(dirs["training_figures"], "confusion_matrix.png")
fig.savefig(fig_path, bbox_inches="tight")

# Precision-recall curves
fig = plot_pr_curves(train_labels, train_preds, test_labels, test_preds)
fig_path = Path(dirs["training_figures"], "pr_curves.png")
fig.savefig(fig_path, bbox_inches="tight")

# Precision-recall threshold curves
fig = plot_pr_threshold_curves(train_labels, train_preds, test_labels, test_preds)
fig_path = Path(dirs["training_figures"], "pr_threshold_curves.png")
fig.savefig(fig_path, bbox_inches="tight")

best_threshold, best_precision = get_threshold(test_labels, test_preds)
logger.info(f"Best Threshold: {best_threshold:.4f}")
logger.info(f"Best Precision: {best_precision:.4f}")

#confusion matrix for best threshold
plot_best_threshold_confusion_matrix(test_labels, test_preds, best_threshold)
logger.info("Figures saved to output folder.")

#%%
logger.info( "Finished running train_and_evaluate.py for:")
logger.info(f"    model_id: {model_id}")
logger.info("----------------------------------------------------------------")
logging.shutdown()
