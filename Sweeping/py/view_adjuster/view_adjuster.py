from abc import abstractmethod, ABC
from collections.abc import Sequence, Iterable

from numpy import ndarray, empty, ravel
from pandas.core.interchange.dataframe_protocol import DataFrame

from model.regression.svregressor import SVRegressor
from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from views.views import Views


class ViewAdjuster(ABC):

    @abstractmethod
    def adjust(self, view_to_adjust: Table, adjusting_views: Views) -> Table:
        raise NotImplementedError()


class FeatureAdjuster(ABC):

    def adjust(self, feature_to_adjust: ndarray, adjusting_views: Views) -> ndarray:
        """feature_to_adjust is a 1-dimensional array. Returns a 1-dimensional array."""
        return self.adjust_df(feature_to_adjust=feature_to_adjust, adjusting_views_df=adjusting_views.to_dataframe())

    @abstractmethod
    def adjust_df(self, feature_to_adjust: ndarray, adjusting_views_df: DataFrame) -> ndarray:
        """feature_to_adjust is a 1-dimensional array. Returns a 1-dimensional array."""
        raise NotImplementedError()


class IterativeViewAdjuster(ViewAdjuster):
    """Iterates on the features to be adjusted."""
    __feature_adjusters: Sequence[FeatureAdjuster]
    """Has one FeatureAdjuster for each feature in the target table."""

    def __init__(self, feature_adjusters: Iterable[FeatureAdjuster]):
        self.__feature_adjusters = list(feature_adjusters)

    def adjust(self, view_to_adjust: Table, adjusting_views: Views) -> Table:
        adjusting_views_df = adjusting_views.to_dataframe()
        np_res = empty(shape=(view_to_adjust.n_row(), view_to_adjust.n_col()), dtype=float)
        for i, f in enumerate(view_to_adjust.np_cols()):
            np_res[:, i] = self.__feature_adjusters[i].adjust_df(
                feature_to_adjust=f, adjusting_views_df=adjusting_views_df)
        return BackedTable(
            backend=NpTable(data=np_res, rownames=view_to_adjust.rownames(), colnames=view_to_adjust.colnames()))


class FeatureAdjusterByRegressor(FeatureAdjuster):
    __regressor: SVRegressor

    def __init__(self, regressor: SVRegressor):
        self.__regressor = regressor

    def adjust_df(self, feature_to_adjust: ndarray, adjusting_views_df: DataFrame) -> ndarray:
        """feature_to_adjust is a 1-dimensional array. Returns a 1-dimensional array."""
        pred = self.__regressor.predict(x=adjusting_views_df)
        pred_flat = ravel(pred)
        if pred_flat.shape != feature_to_adjust.shape:
            raise ValueError(f"Shape mismatch: {feature_to_adjust.shape} vs {pred_flat.shape}")
        return feature_to_adjust - pred_flat
