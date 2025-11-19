import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, average_precision_score

def evaluate_model(model, X_test, y_true, model_name="Classifier", print_c_matrix = True):
    """
    Evaluate a classifier on test data by computing key metrics and displaying a confusion matrix.

    """
    y_pred = model.predict(X_test)

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
