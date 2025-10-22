from sklearn.base import BaseEstimator, TransformerMixin

class DeterministicCleaner(BaseEstimator, TransformerMixin):
    # To inherit method from them eg (.fit) and (.transform)
    """
    """
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
