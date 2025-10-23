import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    # To inherit method from them eg (.fit) and (.transform)
    """
    feature engineering class.

    - Squares potentially strong predictor features.
    - Scales concentrated (low variance) features.
    - Extracts hour and time segment from the Time feature.
    """
    def __init__(self):
        # define features groups
        self.strong_predictors = ["V4", "V9", "V10", "V11", "V12", "V14", "V16", "V17"]
        self.concentrated_predictors = ["V2", "V5", "V7", "V8", "V20","V21", "V23", "V27", "V28", "Amount"]
        self.time_predictor = "Time"
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        # learn the mini and max of the concentrated features
        self.scaler.fit(self.concentrated_predictors)
        return self
    def transform(self, X, y=None):
        X = X.copy()
        # squaring strong predictors
        for column in self.strong_predictors:
            if column in X.columns:
                X[f"{col}_squared"] = X[col] ** 2
        # scale the concentrated columns
        for column in self.concentrated_predictors:
            if column in X.columns:
                X[f"{column}_scaled"] = self.scaler.transform(X[column])





