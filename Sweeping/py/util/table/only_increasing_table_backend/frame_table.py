from typing import Sequence, Union, Optional

from numpy import ndarray
from pandas import DataFrame
from static_frame import Frame

from util.table.only_increasing_table_backend.only_increasing_backend import OnlyIncreasingTableBackend
from util.table.table_backend.np_table import NpTable
from util.table.table_utils import is_table_data, is_2d


class FrameTable(OnlyIncreasingTableBackend):
    """Frame accepts only unique labels for columns and rows. This can be an issue if doing bootstrap."""
    __data: Frame

    def __init__(self, data: Union[Frame, DataFrame, ndarray]):
        if not is_2d(data):
            raise ValueError("data is not 2D")
        if not is_table_data(data):
            raise ValueError("data is not numeric")
        if isinstance(data, ndarray):
            data = Frame(data)
        if isinstance(data, DataFrame):
            data = Frame.from_pandas(data)
        if isinstance(data, Frame):
            self.__data = data
        else:
            raise ValueError("Unsupported input type.")

    def n_row(self) -> int:
        return self.__data.shape[0]

    def n_col(self) -> int:
        return self.__data.shape[1]

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        return NpTable.select_from_numpy(
            data=self.__data.values, selected_rows=selected_rows, selected_cols=selected_cols)

    def to_frame(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> Frame:
        res_np = self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols)
        return Frame(
            data=res_np, index=self.rownames(selected=selected_rows), columns=self.colnames(selected=selected_cols))

    def memory_size(self) -> int:
        return self.size()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        cols = self.__data.columns
        if selected is None:
            return cols
        else:
            return [str(cols[s]) for s in selected]

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        rows = self.__data.index
        if selected is None:
            return rows
        else:
            return [str(rows[s]) for s in selected]

    def has_fast_cols(self) -> bool:
        return True

    def has_fast_rows(self) -> bool:
        return True
