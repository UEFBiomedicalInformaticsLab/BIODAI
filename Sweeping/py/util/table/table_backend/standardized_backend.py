from typing import Sequence, Optional

from numpy import ndarray
from sklearn.preprocessing import StandardScaler

from util.math.feature_standard_scaler import FeatureStandardScalerNumpy, \
    FeatureStandardScaler
from util.table.table_backend.table_backend import TableBackend


class StandardizedBackend(TableBackend):
    """Uses a standard scaler for each column to avoid a slow iteration or rows. The scalers are
    fitted lazily only when needed."""
    __inner: TableBackend
    __scalers: list[Optional[StandardScaler]]
    __scaler: FeatureStandardScaler

    def __init__(self, backend: TableBackend, scaler: FeatureStandardScaler = FeatureStandardScalerNumpy()):
        self.__inner = backend
        self.__scalers = [None for _ in range(self.n_col())]
        self.__scaler = scaler

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        return self.__inner.colnames(selected=selected)

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        return self.__inner.rownames(selected=selected)

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        res = self.__inner.to_new_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
        if selected_cols is None:
            selected_cols = range(self.n_col())
        scalers = self.__scalers
        for i, c in enumerate(selected_cols):
            if scalers[c] is None:
                scalers[c] = self.__scaler.fit(
                    self.__inner.to_numpy(selected_rows=None, selected_cols=[c]).ravel())
            res[:, i] = scalers[c].transform(res[:, i])
        return res

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        from util.table.table_backend.np_table import NpTable
        return NpTable(
            data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
            rownames=self.rownames(selected=selected_rows),
            colnames=self.colnames(selected=selected_cols))

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()

