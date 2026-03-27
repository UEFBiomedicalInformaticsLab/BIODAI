from typing import Sequence, Optional

import numpy as np
from numpy import ndarray
from util.table.renamed_table import RenamedRowsTable, OverwriteRenamingFunction
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from util.table.table_backend.table_backend import TableBackend


class ConcatTable(TableBackend):
    __upper: Table
    __lower: Table

    def __init__(self, upper: Table, lower: Table, reset_index: bool = False):
        if upper.n_col() != lower.n_col():
            raise ValueError()
        if upper.colnames() != lower.colnames():
            raise ValueError()
        if reset_index:
            up_n_row = upper.n_row()
            upper = RenamedRowsTable(
                table=upper, renaming_function=OverwriteRenamingFunction(new_names=[str(i) for i in range(up_n_row)]))
            tot_rows = up_n_row + lower.n_row()
            lower = RenamedRowsTable(
                table=lower,
                renaming_function=OverwriteRenamingFunction(new_names=[str(i) for i in range(up_n_row, tot_rows)]))
        if not set(upper.rownames()).isdisjoint(set(lower.rownames())):
            raise ValueError()
        self.__upper = upper
        self.__lower = lower

    def n_row(self) -> int:
        return self.__upper.n_row() + self.__lower.n_row()

    def n_col(self) -> int:
        return self.__upper.n_col()

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        return self.__upper.select_cols(selected=selected).colnames()

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        res = []
        res.extend(self.__upper.rownames())
        res.extend(self.__lower.rownames())
        if selected is not None:
            res = [res[s] for s in selected]
        return res

    def memory_size(self) -> int:
        return self.__upper.memory_size() + self.__lower.memory_size()

    def to_numpy(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> ndarray:
        up = self.__upper.select_cols(selected=selected_cols)
        low = self.__lower.select_cols(selected=selected_cols)
        if selected_rows is None:
            return np.concatenate((up.to_numpy(), low.to_numpy()), axis=0)
        else:
            up_rows = up.n_row()
            arrays = []
            for r in selected_rows:
                if r < up_rows:
                    arrays.append(up.select_rows(selected=[r]).to_numpy())
                else:
                    arrays.append(low.select_rows(selected=[r-up_rows]).to_numpy())
            return np.concatenate(arrays, axis=0)

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        return NpTable(
            data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
            rownames=self.rownames(selected=selected_rows),
            colnames=self.colnames(selected=selected_cols))

    def has_fast_cols(self) -> bool:
        return self.__upper.has_fast_cols() and self.__lower.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__upper.has_fast_rows() and self.__lower.has_fast_rows()
