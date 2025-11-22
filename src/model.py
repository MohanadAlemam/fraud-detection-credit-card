import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (classification_report,
                             average_precision_score,
                             roc_auc_score,
                             balanced_accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone # to handle CatBoost or other models that sometimes break cloning).
from sklearn.pipeline import Pipeline
from copy import deepcopy


# Out-of-fold (OOF) validation
def oof_validation(model_dict: dict, X, y, categorical_features =None):
    """
    Perform out-of-fold (OOF) evaluation/ validation for multiple classification models.

    :param (dict): Dictionary of model_name: model_object pairs.
    :param (pd.DataFrame): Feature matrix.
    :param (pd.Series): Target labels (binary).

    :return: pd.DataFrame: Concatenated metrics table for all models, including: per-class precision, recall, F1-score, PR AUC, and ROC AUC,
    sorted by PR AUC in descending order.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3479)
    compare_models = []

    # 1. CatBoostClassifier.
    for model_name, model in model_dict.items():
        # Checks and handle CatBoost or Pipelines whose final estimator is CatBoost
        if isinstance(model, Pipeline) and isinstance(model.steps[-1][1], CatBoostClassifier):
            oof_probs = np.zeros(len(y))
            step_name = model.steps[-1][0] # get the classifier name

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]

                est = deepcopy(model)
                est.fit(
                    X_train_fold,
                    y_train_fold,
                    **{f"{step_name}__cat_features": categorical_features,
                       f"{step_name}__verbose": False},
                    # ** Python are used for unpacking dictionaries into keyword arguments
                )
                oof_probs[val_idx] = est.predict_proba(X_val_fold)[:, 1]

        # 2. Other classifiers
        else:
            estimator = clone(model)
            # OOF predicted probabilities
            oof_probs = cross_val_predict(
                estimator,
                X,
                y,
                cv=cv,
                n_jobs=-1,
                method='predict_proba')[:,1] # slice positive class probabilities
        # OOF predictions
        oof_preds = (oof_probs >= 0.5).astype(int) # coverts probabilities to labels

        # Classification report
        report_dict = classification_report(y, oof_preds, output_dict=True)

        # PR AUR, ROC AUCc and balanced accu
        pr_auc = average_precision_score(y, oof_probs)
        roc_auc = roc_auc_score(y, oof_probs)

        # Metric table from a dict
        metrics_df = pd.DataFrame(report_dict).transpose()
        metrics_df.drop(index=["accuracy", "macro avg", "weighted avg"], inplace=True)
        # Drop accuracy from index, we have balanced accuracy instead
        metrics_df.drop(columns=["support"], inplace=True)  # drop support from index, for simplicity
        metrics_df["val pr auc"] = pr_auc
        metrics_df["val roc auc"] = roc_auc

        compare_models.append(metrics_df)

    table = pd.concat(compare_models)
    table = table.sort_values("val pr auc", ascending=False)
    return table.round(3)


# Test Evaluation
def test_evaluation(model, X_test, y_true, model_name="Classifier", print_c_matrix=True):
    """
    Evaluate a classifier on test data by computing key metrics and displaying a confusion matrix.

    :param model: the trained model
    :param X_test: the test data
    :param y_true: the true labels
    :param model_name: the name of the model
    :return: metrics table of balanced accuracy, precision, recall, f1_score, PR AUCc, and ROC AUR. Prints confusion_matrix.
    """
    # y predict and y proba
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Slice the positive-class probability

    # Classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # PR AUR, ROC AUCc and balanced accu
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Metric table from a dic
    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df.drop(index=["accuracy", "macro avg", "weighted avg"], inplace=True)
    # Drop accuracy from index, we have balanced accuracy instead
    metrics_df.drop(columns=["support"], inplace=True) #drop support from index, for simplicity

    metrics_df["balanced accuracy"] = balanced_accuracy
    metrics_df["PR AUC"] = pr_auc
    metrics_df["ROC AUC"] = roc_auc

    metrics_df = metrics_df.round(3)

    classes = np.unique(y_true)
    if print_c_matrix:  # Controls Whether to print the C-matrix or not
        confu_matrix = confusion_matrix(y_true, y_pred)
        confu_matrix_normalized = confu_matrix.astype('float') / confu_matrix.sum(axis=1)[:, np.newaxis]
        # Normalized row-wise shows proportions instead of raw counts.
        # numpy divide each row of the confusion matrix by its row sum.
        # Result each row sums to 1, giving proportions per actual class
        display = ConfusionMatrixDisplay(confusion_matrix=confu_matrix_normalized, display_labels=classes)

        display.plot(cmap="Blues", xticks_rotation=45, values_format= ".3f")
        # values_format=".3f" shows the decimals in the heatmap.
        plt.title(f"{model_name}\n\nTest Data Confusion Matrix")
        plt.show()
    return metrics_df


def compare_test_metrics(model_dict: dict):
    """
    Extracts the last-row metric values from each model's metrics DataFrame and
    returns a comparison DataFrame with models as rows and metrics as columns.

    :param model_dict: dict of model names and their respective metrics
    :return: comparison DataFrame
    """
    comp_dict = {}
    for model_name, metrics in model_dict.items():
        metrics_class1 = metrics.iloc[1] # take class 1 row (positive class)

        comp_dict[model_name] = metrics_class1

    comp_df = pd.DataFrame.from_dict(comp_dict, orient="index")
    comp_df.sort_values("PR AUC", ascending=False, inplace=True)
    return comp_df



