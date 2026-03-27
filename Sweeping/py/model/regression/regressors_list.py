from model.regression.knr_cv import KnrCvModel
from model.regression.mlp_ridge import MLPRidgeModel
from model.regression.pruned_tree import PrunedTree
from model.regression.randomized_svr import RandomizedSVRModel
from model.regression.regressors_library import ZeroRegressor, Linear, Lasso, SVRegressorModel, TreeRegressor, RFRegressor, \
    Ridge, MLPRegressorModel, KNRModel, DummyRegressorModel
from model.regression.svr_ridge import SVRRidgeModel

REGRESSORS = (ZeroRegressor(), DummyRegressorModel(strategy="median"),
              Linear(), Lasso(), Ridge(),
              KNRModel(), KnrCvModel(),
              TreeRegressor(criterion="absolute_error"), PrunedTree(square_error=False),
              RFRegressor(criterion="absolute_error"),
              SVRegressorModel(), SVRRidgeModel(), RandomizedSVRModel(),
              MLPRegressorModel(),
              MLPRidgeModel())
NICK_TO_REGRESSOR = {}
for r in REGRESSORS:
    NICK_TO_REGRESSOR[r.nick()] = r
