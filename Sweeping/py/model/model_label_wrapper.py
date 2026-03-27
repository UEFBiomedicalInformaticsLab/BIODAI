from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder


class ModelLabelWrapper:
    """A wrapper for sklearn models that do not support string outcome labels."""
    __label_encoder: LabelEncoder
    __sklearn_model: Any

    def __init__(self, sklearn_model):
        self.__sklearn_model = sklearn_model
        self.__label_encoder = LabelEncoder()
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y, sample_weight=None):
        # Encode string labels to integers
        y_encoded = self.__label_encoder.fit_transform(y)
        # Now y_encoded contains integers like [0, 1, 2, 3, 4]
        # y_encoded can be used to train your model
        self.__sklearn_model.fit(X, y_encoded, sample_weight=sample_weight)

        # Required by sklearn’s check_is_fitted()
        self.classes_ = self.__label_encoder.classes_
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        return self

    def predict(self, X):
        y_pred = self.__sklearn_model.predict(X)
        return self.__label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        return self.__sklearn_model.predict_proba(X)

    def score(self, X, y):
        y_encoded = self.__label_encoder.transform(y)
        return self.__sklearn_model.score(X, y_encoded)

    @property
    def feature_importances_(self):
        # Only expose if the underlying model has this attribute
        if hasattr(self.__sklearn_model, "feature_importances_"):
            return self.__sklearn_model.feature_importances_
        else:
            raise AttributeError(f"{self.__sklearn_model.__class__.__name__} does not support feature_importances_")
