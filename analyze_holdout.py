#Analyze the future predictions on the hold out data set

import pandas as pd
import numpy as np  
import os
from pathlib import Path
from shpretention.project_utils import get_future_predictions_dir
from shpretention.plot_utils import get_precision_recall, plot_confusion_matrix, get_threshold, plot_best_threshold_confusion_matrix
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                           precision_score, recall_score, roc_curve)


#Retrive future predictions from save location
model_type = "xgboost" # xgboost, svm, random_forest, lgbm
output_root = os.path.join("C:/repos/shp_retention/output/" + model_type)
model_id = "850412e6-6764-484a-af88-436d3b88ed92"
run_date = '2023-11-22'
subdir_prefix=f"run_date={run_date}"
future_predictions_id = '25ba0485-5ea2-4db4-a5ed-d55cc8a05967'

dirs = get_future_predictions_dir(output_root, model_id, subdir_prefix, future_predictions_id)


labels = pd.read_csv(Path(dirs["future_predictions_subdir"], "future_labels.csv"), index_col='features_id')
future_preds = pd.read_csv(Path(dirs["future_predictions_subdir"], "future_preds.csv"), index_col='features_id')
future_features = pd.read_csv(Path(dirs["future_predictions_subdir"], "future_features.csv"), index_col='features_id')

future_fpr, future_tpr, _ = roc_curve(labels, future_preds)

#get AUC, precsion and confusion matrix
auc(future_fpr, future_tpr)
get_precision_recall(labels, future_preds)
plot_confusion_matrix(labels, future_preds)

#set threshold basedo on train/test data
predictions = np.where(future_preds.prediction > 0.6576, 1, 0)
predictions.sum()
precision_score(labels,predictions)

#get best threshold and best precision based on hold out data set
best_threshold, best_precision = get_threshold(labels, future_preds)
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best Precision: {best_precision:.4f}")

#confusion matrix for best threshold
plot_best_threshold_confusion_matrix(labels, future_preds, best_threshold)


