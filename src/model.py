from operator import index

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             average_precision_score,
                             roc_auc_score,
                             balanced_accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

def evaluate_model(model, X_test, y_true, model_name="Classifier", print_c_matrix=True):
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
    dic = classification_report(y_true, y_pred, output_dict=True)

    # PR AUR, ROC AUCc and balanced accu
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Metric table from a dic
    metrics_df = pd.DataFrame(dic).transpose()
    metrics_df.drop(index=["accuracy", "macro avg", "weighted avg"], inplace=True)
    # Drop accuracy from index, we have balanced accuracy instead
    metrics_df.drop(columns=["support"], inplace=True) #drop support from index, for simplicity

    metrics_df["Balanced Accuracy"] = balanced_accuracy
    metrics_df["PR AUC"] = pr_auc
    metrics_df["ROC AUC"] = roc_auc

    metrics_df = metrics_df.round(3)
    classes = np.unique(y_true)

    if print_c_matrix:  # Controls Whether to print the C-matrix or not
        confu_matrix = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=confu_matrix, display_labels=classes)

        display.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"{model_name}\n\nTest Data Confusion Matrix")
        plt.show()
    return metrics_df

def compare_models(model_dict: dict):
    """
    Extracts the last-row metric values from each model's DataFrame and
    returns a comparison DataFrame with models as rows and metrics as columns.

    :param model_dict: dict of model names and their respective metrics
    :return: comparison DataFrame
    """
    comp_dict = {}
    for model_name, metrics in model_dict.items():
        metrics_class1 = metrics.iloc[1] # take class 1 row (positive class)

        comp_dict[model_name] = metrics_class1

    comp_df = pd.DataFrame.from_dict(comp_dict, orient="index")

    return comp_df