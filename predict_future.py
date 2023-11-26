# Predict future outcomes

import logging
import numpy as np
import pandas as pd
import pickle
import os

from shpretention.plot_utils import plot_future_predictions_distribution
from shpretention.project_utils import (set_project_dir, 
                                        create_new_future_predictions_dir, 
                                        get_basic_logger)
from shpretention.data_utils import get_features
from pathlib import Path
from uuid import uuid4

set_project_dir(project="shp_retention")

#%% Configure project details
model_type = "xgboost" # xgboost, svm, random_forest, lgbm
output_root = os.path.join("C:/repos/shp_retention/output/" + model_type)
dsn = "AWSDW"
run_date = "2023-11-22"
window_start = "2022-01-01"
window_stop = "2022-08-31"
model_id = "850412e6-6764-484a-af88-436d3b88ed92"

#%% Set up directories and create logger

# Create new output sub-folder for future predictions
future_predictions_id, dirs = create_new_future_predictions_dir(
    output_root_path=output_root, 
    model_id=model_id, 
    subdir_prefix=f"run_date={run_date}"
)

# Create logger that saves to output folder
logger = get_basic_logger(dirs["log"])

logger.info("----------------------------------------------------------------")
logger.info("Started running predict_future_outcomes.py:")
logger.info(f"    output_root: {output_root}")
logger.info(f"    dsn: {dsn}")
logger.info(f"    run_date: {run_date}")
logger.info(f"    model_id: {model_id}")
logger.info(f"    future_predictions_id: {future_predictions_id}")


#%% Load features and labels

FEATURES = pd.read_csv(Path(dirs["training_data"], "FEATURES.csv"))
FEATURE_TYPES = FEATURES.set_index("type")["name"]
cat_features = FEATURE_TYPES.loc[["nominal"]].tolist() 

#%% Get features

future_features = get_features(
    logger=logger, 
    dsn=dsn, 
    window_start=window_start,
    window_stop=window_stop,
    cat_features=cat_features
)

# Create a surrogate key for the features
future_features.insert(
    0,
    "features_id",
    future_features.index.to_series().map(lambda _: str(uuid4())),
)
future_features = future_features.set_index("features_id")
labels = future_features["quit_plan"]
future_features = future_features.drop("quit_plan", axis=1)
logger.info("DataFrame created:")
logger.info(f"    future_features with shape of {future_features.shape}.")


#%% Load model

pipeln = pickle.load(open(dirs["model_file"], "rb"))


#%% Get future predictions

future_preds = pd.DataFrame(
    data={
        "features_id": future_features.reset_index()["features_id"],
        "prediction": pipeln.predict_proba(future_features)[:, 1]
    }
).set_index("features_id")

logger.info("Prediction quantiles:")
logger.info( "    Data:   Future")
logger.info(f"    Min:    {float(future_preds.min()):.4f}")
logger.info(f"    Q1:     {float(future_preds.quantile(0.25)):.4f}")
logger.info(f"    Median: {float(future_preds.quantile(0.50)):.4f}")
logger.info(f"    Q3:     {float(future_preds.quantile(0.75)):.4f}")
logger.info(f"    Max:    {float(future_preds.max()):.4f}")


#%% Save features and predictions

# Save features
future_features[FEATURES.name].to_csv(Path(dirs["future_predictions_subdir"], "future_features.csv"), index=True)

# Save unused features
if np.any(~future_features.columns.isin(FEATURES.name)):
    future_features.iloc[:,~future_features.columns.isin(FEATURES.name)].to_csv(Path(dirs["future_predictions_subdir"], "unused_features.csv"), index=True)

# Save predictions
future_features = future_features.merge(labels, on='features_id', how='inner')
labels.to_csv(Path(dirs["future_predictions_subdir"], "future_labels.csv"), index=True)
future_preds.to_csv(Path(dirs["future_predictions_subdir"], "future_preds.csv"), index=True)
features_labels_preds = future_features.merge(future_preds, on='features_id', how='inner')
features_labels_preds.to_csv(Path(dirs["future_predictions_subdir"], "features_labels_preds.csv"), index=True)
logger.info("Features and predictions saved to output folder.")


#%% Create and save figures 

# Prediction distribution
fig = plot_future_predictions_distribution(future_preds)
fig_path = Path(dirs["future_predictions_subdir"], "future_predictions_distribution.png")
fig.savefig(fig_path, bbox_inches="tight")

logger.info("Figures saved to output folder.")


#%%
logger.info( "Finished running predict_future_outcomes.py for:")
logger.info(f"    model_id: {model_id}")
logger.info(f"    future_predictions_id: {future_predictions_id}")
logger.info("----------------------------------------------------------------")
logging.shutdown()


#%%

prediction_df = future_features.join(future_preds).reset_index().drop('features_id', axis=1)
final_output = prediction_df[["shp_member_id","age","county_name","state_code","prediction"]]
file_name = run_date + ' ' + model_id
final_output.to_csv(r'C:\tfs\Analytics Services\DataScience\SHP-Retention\shpretention_project\data\''+ file_name +'.csv')

prediction_df.sort_values('prediction',ascending=False,inplace=True)

