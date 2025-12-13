import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.conftest import devices
from sklearn.metrics import (classification_report,
                             average_precision_score,
                             balanced_accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_recall_curve
                             )
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone # to handle CatBoost or other models that sometimes break cloning).
from sklearn.pipeline import Pipeline
from copy import deepcopy

from tensorflow.python.trackable.trackable_utils import escape_local_name


# 1. Out-of-fold (OOF) validation
def oof_validation(model_dict: dict, X, y, categorical_features =None):
    """
    Perform out-of-fold (OOF) evaluation/ validation for multiple classification models.

    :param (dict): Dictionary of model_name: model_object pairs.
    :param (pd.DataFrame): Feature matrix.
    :param (pd.Series): Target labels (binary).

    :return: pd.DataFrame: Concatenated metrics table for all models, including: per-class precision, recall, F1-score, PR AUC, and ROC AUC,
    sorted by PR AUC in descending order.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=3479)
    # 3 separate train - predict cycles per model
    compare_models = []

    for model_name, model in model_dict.items():

        # 1. CatBoostClassifier
        if isinstance(model, Pipeline) and isinstance(model.steps[-1][1], CatBoostClassifier):
            oof_probs = np.zeros(len(y))
            step_name = model.steps[-1][0]

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]

                # rebuild pipeline copy transformers, create fresh CatBoost
                transformer_steps = [(n, deepcopy(estimator)) for n, estimator in model.steps[:-1]]
                orig_catboost = model.steps[-1][1]
                fresh_catboost = CatBoostClassifier(**orig_catboost.get_params())
                estimator = Pipeline(transformer_steps + [(step_name, fresh_catboost)])

                # set GPU
                estimator.set_params(**{f"{step_name}__task_type": "GPU", f"{step_name}__devices": "0"})

                # fit with categorical features
                estimator.fit(
                    X_train_fold,
                    y_train_fold,
                    **{f"{step_name}__cat_features": categorical_features,
                       f"{step_name}__verbose": False #  # suppress iteration output
                       }
                )
                # predict probabilities
                oof_probs[val_idx] = estimator.predict_proba(X_val_fold)[:, 1]

        # 2. Other classifiers
        else:
            estimator = clone(model)
            # OOF predicted probabilities
            oof_probs = cross_val_predict(
                estimator,
                X, y,
                cv=cv,
                n_jobs=-1,
                method='predict_proba')[:,1] # slice positive class probabilities
        # OOF predictions
        oof_preds = (oof_probs >= 0.5).astype(int) # coverts probabilities to labels

        # Classification report
        report_dict = classification_report(y, oof_preds, output_dict=True)

        # PR AUR, ROC AUCc and balanced accu
        pr_auc = average_precision_score(y, oof_probs)

        # Metric table from a dict
        metrics_df = pd.DataFrame(report_dict).transpose()
        # Keep only positive class
        metrics_df = metrics_df.loc[['1']]  # only class 1

        metrics_df["oof pr auc (fraud)"] = pr_auc
        # Add model name column
        metrics_df["model"] = model_name

        compare_models.append(metrics_df)

    table = pd.concat(compare_models, ignore_index=True)
    table.set_index("model", inplace=True)
    table = table.sort_values("oof pr auc (fraud)", ascending=False)
    return table.round(3)


# 2. Test Evaluation
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
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Metric table from a dic
    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df.drop(index=["accuracy", "weighted avg"], inplace=True) # maybe keep macro avg for results
    # Drop accuracy from index, we have balanced accuracy instead
    # Drop, "weighted avg" Weighted average (in imbalanced fraud setting) the positive negative class will skew the results
    #metrics_df.drop(columns=["support"], inplace=True) #drop support from index, for simplicity

    metrics_df["balanced accuracy"] = balanced_accuracy
    metrics_df["pr auc"] = pr_auc
    metrics_df = metrics_df.rename(index={"0": "Non-fraud (Class 0)", "1": "Fraud (Class 1)"}) # Rename the classes

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


# 3. Test metrics Comparison
def compare_test_metrics(model_dict: dict, X_test, y_test):
    """
    Calculates test metrics and extracts positive class metric values from each model's metrics DataFrame and
    returns a rank-ordered table based on PR AUC for positive class.

    :param model_dict: dict of model names and model object
    :return: comparison DataFrame
    """
    comp_dict = {}
    for model_name, model in model_dict.items():
        metrics_df = test_evaluation(model, X_test= X_test,
                                     y_true= y_test,
                                     model_name = model_name,
                                     print_c_matrix=False)

        metrics_class1 = metrics_df.loc["Fraud (Class 1)"] # take class 1 row (positive class)
        comp_dict[model_name] = metrics_class1

    comp_df = pd.DataFrame.from_dict(comp_dict, orient="index")
    comp_df.sort_values("pr auc", ascending=False, inplace=True)
    return comp_df


# 4. Results analysis methods and functions

# Precision-Recall curve (PR AUC)
def pr_curve_oof(model, X, y):
    """
    Compute precision, recall and PR AUC scores for a classifier. and plots the PR-AUC curve.

    :param model: the trained model
    :param X: the test/ validation data
    :param y: the true labels
    :return: precision, recall plot and PR AUC score
    """
    # predict  off probabilities using the model
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=3479)
    oof_y_probabilities = cross_val_predict(model,
                                            X, y,
                                            cv=cv,
                                            n_jobs=-1,
                                            method='predict_proba')[:,1]  # positive class probabilities

    precisions, recalls, _ = precision_recall_curve(y, oof_y_probabilities)
    pr_auc = average_precision_score(y, oof_y_probabilities)

    plt.figure(figsize = (8, 6))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve (PR AUC = {pr_auc:.3f})")
    plt.show()

    return pr_auc, oof_y_probabilities


# Precision & Recall vs Threshold (use the probabilities computed once)
def pr_vs_threshold_curve_oof(y, oof_y_probabilities):
    """
    Plots precision and recall curve of model on test data.

    """
    precision, recall, thresholds  = precision_recall_curve(y, oof_y_probabilities)

# plot precision and recall as the threshold changes
    plt.figure(figsize = (10,5))
    plt.plot(thresholds, precision[:-1], label="Precision") # precision_recall_curve returns values with extra elements, we need to slice
    plt.plot(thresholds, recall[:-1], label="Recall")

    #plt.vlines(thresholds[::50], ymin=0, ymax=1, linestyles='dotted', label="Threshold", alpha=0.5)
#thresholds[::2] plot evey 10th  threshold for clarity
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Figure: Precision-Recall vs. Confidence Threshold")

    plt.show()


# Tiered Thresholds validation / justification

def validate_thresholds(oof_y_probabilities,
                      high_risk_cut_off=0.80,
                      medium_risk_cut_off=0.30):
    """
    Check of how many instances fall into High / Medium / Low tiers
    based on OOF probabilities.
    """

    total = len(oof_y_probabilities)

    high = (oof_y_probabilities >= high_risk_cut_off).sum()
    medium = ((oof_y_probabilities < high_risk_cut_off) &
              (oof_y_probabilities >= medium_risk_cut_off)).sum()
    low = (oof_y_probabilities < medium_risk_cut_off).sum()

    pct_high = high / total * 100
    pct_medium = medium / total * 100
    pct_low = low / total * 100

    pct_df = pd.DataFrame({
        "Percent": [pct_high, pct_medium, pct_low]},
        index = ["High", "Medium", "Low"]
    )
    return pct_df.round(2)

