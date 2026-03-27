from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class ProbabilityWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, base_model, method='sigmoid', cv=5):
        """
        Wraps a classifier to add probability support using CalibratedClassifierCV.

        Parameters:
        - base_model: scikit-learn classifier (must support decision_function or predict_proba)
        - method: 'sigmoid' (Platt scaling) or 'isotonic'
        - cv: number of cross-validation folds or a cross-validation strategy
        """
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.calibrated_model = None
        self.classes_ = None  # Will be set during fit

    def fit(self, X, y, sample_weight=None):
        self.calibrated_model = CalibratedClassifierCV(self.base_model, method=self.method, cv=self.cv)
        self.calibrated_model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.calibrated_model.classes_  # Set classes_ after fitting
        return self

    def predict(self, X):
        check_is_fitted(self.calibrated_model)
        return self.calibrated_model.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self.calibrated_model)
        return self.calibrated_model.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self.calibrated_model)
        return self.calibrated_model.decision_function(X)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self.calibrated_model)
        return self.calibrated_model.score(X, y, sample_weight=sample_weight)

