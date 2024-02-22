from model.regression.pruned_tree import PrunedTree
from model.regression.randomized_svr import RandomizedSVRModel
from model.regression.regressors_library import ZeroRegressor, Linear, Lasso, SVRegressor, TreeRegressor, RFRegressor, \
    Ridge, MLPRegressorModel, KNRModel, DummyRegressorModel
from model.regression.svr_ridge import SVRRidgeModel

REGRESSORS = (ZeroRegressor(), DummyRegressorModel(strategy="median"),
              Linear(), Lasso(), Ridge(),
              KNRModel(),
              TreeRegressor(criterion="absolute_error"), PrunedTree(square_error=False),
              RFRegressor(criterion="absolute_error"),
              SVRegressor(), SVRRidgeModel(), RandomizedSVRModel(),
              MLPRegressorModel())
NICK_TO_REGRESSOR = {}
for r in REGRESSORS:
    NICK_TO_REGRESSOR[r.nick()] = r
