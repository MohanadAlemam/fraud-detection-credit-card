import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    # To inherit method from them eg (.fit) and (.transform)
    """
    Feature engineering class.
    - Absolute value transformation for strong predictors.
    - Squared transformation for the four strongest predictors (V4, V11, V12, V14).
    - Scaling of concentrated (low variance) features.
    - Extract hour and time segment from Time.
    """
    def __init__(self):
        # define features groups
        self.strong_predictors = ["V4", "V9", "V10", "V11", "V12", "V14", "V16", "V17"]
        self.strongest_4predictors = ["V4", "V11", "V12","V14"]
        # These four features exhibit the strongest interquartile range separation between fraud and non-fraud transactions.
        self.concentrated_predictors = ["V2", "V5", "V7", "V8", "V20","V21", "V23", "V27", "V28", "Amount"]
        self.time_predictor = "Time"
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):

        # learn the mini and max of the concentrated features
        self.scaler.fit(X[self.concentrated_predictors])
        return self
    def transform(self, X, y=None):
        X = X.copy()

        # strong predictors
        for column in self.strong_predictors:
            if column in X.columns:
                X[f"{column}_abs"] = X[column].abs()
       # Sq for the 4 strongest predictors
        for column in self.strongest_4predictors:
            if column in X.columns:
                X[f"{column}_sq"] = X[column] ** 2

        # scale the concentrated columns using the MinMaxScaler fitted on training data
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.concentrated_predictors]), # apply the same scaling learned in fit()
            columns=[f"{col}_scaled" for col in self.concentrated_predictors],
            index=X.index # keep the original row order
        )
        X = pd.concat([X, X_scaled], axis=1)  # add the scaled columns to the dataset

        # extract hour and time from feature 'Time'
        if self.time_predictor in X.columns:
            X["Hour_of_day"] = (X["Time"] % 86400) // 3600
            # modulo operation ensures the 24-hour cycle resets, 3600 convert seconds to hours
            # 86400 = number of seconds in a day ie 24 hrs
            # 3600 = number of seconds in one hour
            X["Time_segment"] = pd.cut(X["Hour_of_day"],
                                       bins = [0, 8, 16, 24],
                                       labels=["early_morning","morning_afternoon", "evening_night"],
                                       #  0–7 hours, 8–15 hours, 16–23 hours
                                       right=False)
        X.drop(columns = self.concentrated_predictors + ["Time"], inplace = True)
        # drop these column as the transformed versions curry the signal in a better way
        return X