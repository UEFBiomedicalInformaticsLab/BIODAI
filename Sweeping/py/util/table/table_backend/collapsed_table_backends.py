from collections.abc import Iterable
from typing import Sequence, Optional

import numpy as np
from numpy import ndarray

from multi_view_utils import prefix_a_col, prefix_all
from util.table.collapsed_tables import TableAssignment
from util.table.table_backend.np_table import NpTable
from util.table.table_backend.table_backend import TableBackend


class CollapsedTableBackends(TableBackend):
    __tables: Sequence[TableBackend]

    def __init__(self, tables: Sequence[TableBackend]):
        if len(tables) == 0:
            self.__tables = []
        else:
            self.__tables = list(tables)
            nrows = self.__tables[0].n_row()
            for i in range(1, len(self.__tables)):
                if self.__tables[i].n_row() != nrows:
                    raise ValueError("All tables must have the same number of rows.")

    def n_row(self) -> int:
        if len(self.__tables) > 0:
            return self.__tables[0].n_row()
        else:
            return 0

    def n_col(self) -> int:
        res = 0
        for t in self.__tables:
            res += t.n_col()
        return res

    def to_numpy(self, selected_rows: Sequence[int], selected_cols: Sequence[int]) -> ndarray:
        if len(self.__tables) == 0:
            return np.zeros(shape=(0, 0))
        else:
            arrays = [self.__tables[a.table].to_numpy(selected_rows=selected_rows, selected_cols=(a.column,))
                      for a in self.__cols_assign(selected=selected_cols)]
            return np.concatenate(arrays, axis=1)

    def compile(self, selected_rows: Optional[Sequence[int]], selected_cols: Optional[Sequence[int]]) -> TableBackend:
        return NpTable(data=self.to_numpy(selected_rows=selected_rows, selected_cols=selected_cols),
                       rownames=self.rownames(selected=selected_rows), colnames=self.colnames(selected=selected_cols))

    def memory_size(self) -> int:
        res = 0
        for t in self.__tables:
            res += t.memory_size()
        return res

    def rownames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if len(self.__tables) == 0:
            return []
        else:
            return self.__tables[0].rownames(selected=selected)

    def colnames(self, selected: Optional[Sequence[int]]) -> Sequence[str]:
        if selected is None:
            res = []
            for i, t in enumerate(self.__tables):
                res.extend(prefix_all(view_index=i, col_names=t.colnames(selected=None)))
        else:
            return [
                prefix_a_col(view_index=a.table, col_name=self.__tables[a.table].colnames(selected=(a.column,))[0])
                for a in self.__cols_assign(selected=selected)]

    def __cols_assign(self, selected: Sequence[int]) -> Sequence[TableAssignment]:
        if not self.cols_in_range(selected=selected):
            raise ValueError("Column out of range.")
        return self.cols_assign_static(tables_n_col=[t.n_col() for t in self.__tables], selected=selected)

    @staticmethod
    def cols_assign_static(tables_n_col: Sequence[int], selected: Iterable[int]) -> Sequence[TableAssignment]:
        cum_sizes = []
        tot = 0
        for t in tables_n_col:
            cum_sizes.append(tot + t)
        res = []
        for i in selected:
            table_index = 0
            s = cum_sizes[table_index]
            while i >= s:
                table_index += 1
                s = cum_sizes[table_index]
            if table_index == 0:
                res.append(TableAssignment(table=table_index, column=i))
            else:
                res.append(TableAssignment(table=table_index, column=i - cum_sizes[table_index - 1]))
        return res

    def has_fast_cols(self) -> bool:
        for t in self.__tables:
            if not t.has_fast_cols():
                return False
        return True

    def has_fast_rows(self) -> bool:
        for t in self.__tables:
            if not t.has_fast_rows():
                return False
        return True
