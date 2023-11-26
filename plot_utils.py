#plot_utils.py
#helper functions for plotting

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from shpretention.model_utils import get_feature_importances, get_importances
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

blue = (0.39215686274509803, 0.70980392156862748, 0.80392156862745101)


def _combine_train_test_preds(train_labels, train_preds, test_labels, test_preds):
    if (test_labels is not None) or (test_preds is not None):
        return pd.concat(
            [
                pd.DataFrame({"label": train_labels.iloc[:,0], "prediction": train_preds.iloc[:,0]}).assign(
                    result_set="train"
                ),
                pd.DataFrame({"label": test_labels.iloc[:,0], "prediction": test_preds.iloc[:,0]}).assign(
                    result_set="test"
                )
            ]
        )
    else:
        return [
            pd.DataFrame({"label": "future", "prediction": train_preds.iloc[:,0]}).assign(
                result_set="future"
            )
        ]


def plot_coefficients(FEATURE_TYPES, pipeline):
    # coefficients = get_coefficients(FEATURE_TYPES, pipeline)
    raise RuntimeError("plot_utils.plot_coefficients() has not been implemented")

def plot_perm_importances(FEATURE_TYPES, pipeline, test_features, test_labels):
    feature_importances = get_importances(FEATURE_TYPES, pipeline, test_features, test_labels)
    fig, ax = plt.subplots()
    ax = sns.barplot(
        x="Importance", 
        y="Feature", 
        data=feature_importances.head(25), 
        color=blue, 
        ax=ax
    )
    ax.set_title("Top 25 Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")

    return fig


def plot_feature_importance(FEATURE_TYPES, pipeline):
    feature_importances = get_feature_importances(FEATURE_TYPES, pipeline)
    fig, ax = plt.subplots()
    ax = sns.barplot(
        x="Importance", 
        y="Feature", 
        data=feature_importances.head(25), 
        color=blue, 
        ax=ax
    )
    ax.set_title("Top 25 Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")

    return fig


def plot_prediction_distributions(train_labels, train_preds, test_labels, test_preds, hue="label"):
    results = _combine_train_test_preds(train_labels, train_preds, test_labels, test_preds)

    fig = sns.FacetGrid(
        results,
        hue=hue,
        col="result_set",
        sharex=True,
        sharey=False,
        height=4,
        aspect=2
    )
    fig.map(sns.histplot, "prediction", bins=np.linspace(0,1,51), kde=False, stat="density", common_norm=False, edgecolor="none")
    fig.axes[0][0].set_title("Train Prediction Distribution")
    fig.axes[0][0].set_xlim([0,1])    
    fig.axes[0][1].set_title("Test Prediction Distribution")
    fig.axes[0][1].set_xlim([0,1])    
    fig.set_ylabels("Density")
    fig.set_xlabels("Prediction")
    fig.add_legend()
    return fig


def plot_future_predictions_distribution(future_preds):
    fig, ax = plt.subplots(1,1, figsize=[4,4])
    sns.histplot(future_preds["prediction"], bins=np.linspace(0,1,51), kde=False, stat="density", edgecolor="none", ax=ax)
    ax.set_title("Future Predictions Distribution")
    ax.set_xlim([0,1])
    ax.set_ylabel("Density")
    ax.set_xlabel("Prediction")
    return fig


def plot_roc_curves(train_labels, train_preds, test_labels, test_preds):
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_preds)
    train_roc_auc = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, _ = roc_curve(test_labels, test_preds)
    test_roc_auc = auc(test_fpr, test_tpr)

    roc_aucs = pd.concat(
        [
            pd.DataFrame({"fpr": train_fpr, "tpr": train_tpr}).assign(roc_auc_set="train"),
            pd.DataFrame({"fpr": test_fpr, "tpr": test_tpr}).assign(roc_auc_set="test"),
        ]
    )

    fig = sns.FacetGrid(roc_aucs, col="roc_auc_set", height=5, aspect=1)
    fig.map(sns.lineplot, "fpr", "tpr", ci=None)
    fig.set_axis_labels("False Positive Rate", "True Positive Rate")

    fig.axes[0][0].set_title("Train ROC Curve")
    fig.axes[0][0].legend(loc="lower right", labels=["AUC = %0.4f" % train_roc_auc])
    fig.axes[0][0].plot([0, 1], [0, 1], "r--")

    fig.axes[0][1].set_title("Test ROC Curve")
    fig.axes[0][1].legend(loc="lower right", labels=["AUC = %0.4f" % test_roc_auc])
    fig.axes[0][1].plot([0, 1], [0, 1], "r--")

    return (fig, train_roc_auc, test_roc_auc)


def plot_pr_curves(train_labels, train_preds, test_labels, test_preds):
    train_precision, train_recall, train_thresholds = precision_recall_curve(
        train_labels, train_preds
    )
    train_thresholds = np.insert(train_thresholds, 0, 0)
    train_pr_auc = auc(train_recall, train_precision)

    test_precision, test_recall, test_thresholds = precision_recall_curve(test_labels, test_preds)
    test_thresholds = np.insert(test_thresholds, 0, 0)
    test_pr_auc = auc(test_recall, test_precision)

    pr_aucs = pd.concat(
        [
            pd.DataFrame(
                {
                    "recall": train_recall,
                    "precision": train_precision,
                    "threshold": train_thresholds,
                }
            ).assign(pr_auc_set="train"),
            pd.DataFrame(
                {
                    "recall": test_recall,
                    "precision": test_precision,
                    "threshold": test_thresholds,
                }
            ).assign(pr_auc_set="test"),
        ]
    )

    fig = sns.FacetGrid(pr_aucs, col="pr_auc_set", height=5, aspect=1)
    fig.map(sns.lineplot, "recall", "precision", ci=None)
    fig.set_axis_labels("Recall", "Precision")

    fig.axes[0][0].set_title("Train Precision/Recall Curve")
    fig.axes[0][0].legend(loc="upper right", labels=["AUC = %0.4f" % train_pr_auc])

    fig.axes[0][1].set_title("Test Precision/Recall Curve")
    fig.axes[0][1].legend(loc="upper right", labels=["AUC = %0.4f" % test_pr_auc])

    return fig


def plot_pr_threshold_curves(train_labels, train_preds, test_labels, test_preds):
    train_precision, train_recall, train_thresholds = precision_recall_curve(
        train_labels, train_preds
    )
    train_thresholds = np.insert(train_thresholds, 0, 0)

    test_precision, test_recall, test_thresholds = precision_recall_curve(test_labels, test_preds)
    test_thresholds = np.insert(test_thresholds, 0, 0)

    pr_aucs = pd.concat(
        [
            pd.DataFrame(
                {
                    "recall": train_recall,
                    "precision": train_precision,
                    "threshold": train_thresholds,
                }
            ).assign(pr_auc_set="train"),
            pd.DataFrame(
                {
                    "recall": test_recall,
                    "precision": test_precision,
                    "threshold": test_thresholds,
                }
            ).assign(pr_auc_set="test"),
        ]
    )

    fig = sns.FacetGrid(
        pd.melt(
            pr_aucs.loc[pr_aucs.threshold > 0],
            id_vars=["threshold", "pr_auc_set"],
            var_name="metric",
        ),
        hue="metric",
        col="pr_auc_set",
        height=5,
        aspect=1,
    )
    fig.map(sns.lineplot, "threshold", "value", ci=None)
    fig.set_axis_labels("Threshold", "Precision/Recall")
    fig.axes[0][0].set_title("Train Threshold Curves")
    fig.axes[0][1].set_title("Test Threshold Curves")
    fig.add_legend()

    return fig


def plot_confusion_matrix(test_labels, test_preds):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig, axs = plt.subplots(3, 3, constrained_layout=True)
    fig.set_size_inches(10, 10)
    axs = axs.reshape((9,))

    for i in range(len(thresholds)):
        preds = (test_preds > thresholds[i]).astype(int)
        cm = confusion_matrix(test_labels, preds)
        ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", cbar=False, ax=axs[i])
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"Threshold = {thresholds[i]}")
        ax.xaxis.set_ticklabels(["negative", "positive"])
        ax.yaxis.set_ticklabels(["negative", "positive"])

    return fig

def plot_best_threshold_confusion_matrix(test_labels, test_preds, best_threshold):
    preds = (test_preds > best_threshold).astype(int)
    cm = confusion_matrix(test_labels, preds)
    plt.figure(figsize=(8,6), dpi=100)
    # Scale up the size of all text
    sns.set(font_scale = 1.1)
    ax = sns.heatmap(cm, annot=True, fmt='d', )

    # set x-axis label and ticks. 
    ax.set_xlabel("Predicted to Disenroll", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])

    # set y-axis label and ticks
    ax.set_ylabel("Actually Disenrolled ", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])

    # set plot title
    ax.set_title("Confusion Matrix for the Xgboost retention Model", fontsize=14, pad=20)

    plt.show()



def get_precision_recall(test_labels, test_preds):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df = pd.DataFrame(data={"threshold": thresholds})

    def _set_pr(row):
        preds = (test_preds > row["threshold"]).astype(int)
        row["precision"] = precision_score(test_labels, preds, average="binary")
        row["recall"] = recall_score(test_labels, preds, average="binary")
        return row

    return df.apply(_set_pr, axis=1)


def get_threshold(test_labels, test_preds):
    precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
    fscore = (2 * precision * recall) / (precision + recall)
    fmax = np.argmax(fscore)
    return thresholds[fmax], fscore[fmax]

