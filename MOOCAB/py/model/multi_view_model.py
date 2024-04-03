from __future__ import annotations

from model.mv_predictor import MVPredictor


# Abstract class for models able to learn creating a predictor.
class MVModel:

    # x is a list of views, each being a matrix with samples on rows.
    # y is a list of anything, each element being an expected output.
    # Returns a Predictor
    def fit(self, views, y) -> MVPredictor:
        raise NotImplementedError()

    def fit_and_predict(self, views_train, y_train, views_test):
        predictor = self.fit(views=views_train, y=y_train)
        predictions_on_train = predictor.predict(views=views_train)
        predictions_on_test = predictor.predict(views=views_test)
        return predictions_on_train, predictions_on_test


# A model that has tunable hyperparameters.
class MVTunableModel:

    def tune(self, hyperparameters) -> MVModel:
        raise NotImplementedError()
