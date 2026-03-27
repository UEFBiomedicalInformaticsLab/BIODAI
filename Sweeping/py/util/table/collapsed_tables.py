from typing import Sequence, Optional, Iterable

import numpy as np
import pandas
from frozenlist import FrozenList
from numpy import ndarray
from pandas import DataFrame, Index

from multi_view_utils import view_prefix
from util.sequence_utils import equal_iterables
from util.table.renamed_table import RenamedColsTable, PrefixRenamingFunction
from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS


class TableAssignment:
    table: int
    column: int

    def __init__(self, table: int, column: int):
        self.table = table
        self.column = column

    def __str__(self) -> str:
        return "(" + str(self.table) + ", " + str(self.column) + ")"


class CollapsedTables(Table):
    __tables: Sequence[Table]
    __table_assignments: FrozenList[TableAssignment]
    # For each column index the table from which to extract the column and the index of the column in that table.

    def __init__(self, tables: Sequence[Table], table_assignments: Optional[Sequence[TableAssignment]] = None,
                 check_rownames: bool = False):
        """table_assignments:
        for each column index the table from which to extract the column and the index of the column in that table.
        Pass RenamedTables to have different name schemas for the different tables."""
        n_tables = len(tables)
        if n_tables == 0:
            self.__tables = []
        else:
            self.__tables = list(tables)
            if n_tables > 1:
                nrows = self.__tables[0].n_row()
                for i in range(1, len(self.__tables)):
                    if self.__tables[i].n_row() != nrows:
                        raise ValueError("All tables must have the same number of rows.")
                if check_rownames:
                    rownames = self.__tables[0].rownames()
                    for i in range(1, len(self.__tables)):
                        if not equal_iterables(rownames, self.__tables[i].rownames()):
                            raise ValueError("All tables must have the same row names.")
        if table_assignments is None:
            assign = FrozenList()
            for t in range(n_tables):
                for i in range(self.__tables[t].n_col()):
                    assign.append(TableAssignment(table=t, column=i))
            self.__table_assignments = assign
        else:
            if isinstance(table_assignments, FrozenList):
                self.__table_assignments = table_assignments
            else:
                self.__table_assignments = FrozenList(table_assignments)
        if not self.__n_col_by_tables() == len(self.__table_assignments):
            raise ValueError()
        self.__table_assignments.freeze()

    def n_row(self) -> int:
        if len(self.__tables) > 0:
            return self.__tables[0].n_row()
        else:
            return 0

    def n_col(self) -> int:
        return len(self.__table_assignments)

    def __n_col_by_tables(self) -> int:
        res = 0
        for t in self.__tables:
            res += t.n_col()
        return res

    def to_numpy(self) -> ndarray:
        if len(self.__tables) == 0:
            return np.zeros(shape=(0, 0))
        elif len(self.__table_assignments) == 0:
            return np.zeros(shape=(self.n_row(), 0))
        else:
            arrays = [self.__tables[ta.table].select_cols(selected=(ta.column,)).to_numpy()
                      for ta in self.__table_assignments]
            return np.concatenate(arrays, axis=1)

    def inner_to_dataframe(self) -> DataFrame:
        if len(self.__tables) == 0:
            return DataFrame(np.zeros(shape=(0, 0)))
        elif len(self.__table_assignments) == 0:
            return DataFrame(np.zeros(shape=(self.n_row(), 0)), index=Index(self.rownames()))
        else:
            col_names = self.colnames()
            frames = []
            for ta, cn in zip(self.__table_assignments, col_names):
                df = self.__tables[ta.table].select_cols(selected=(ta.column,)).to_dataframe()
                df = df.set_axis(labels=[cn], axis=1, copy=False)  # We need to use the new labels to avoid duplicated names.
                frames.append(df)
            res = pandas.concat(objs=frames, axis=1, copy=False, ignore_index=False)
            return res

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            from util.table.backed_table import BackedTable
            from util.table.table_backend.np_table import NpTable
            return BackedTable(
                backend=NpTable(data=self.to_numpy(), rownames=self.rownames(), colnames=self.colnames()))

    def memory_size(self) -> int:
        res = 0
        for t in self.__tables:
            res += t.memory_size()
        return res

    def select_rows(self, selected: Iterable[int]) -> Table:
        res_tables = []
        for t in self.__tables:
            res_tables.append(t.select_rows(selected=selected))
        return CollapsedTables(tables=res_tables, table_assignments=self.__table_assignments)

    def select_cols(self, selected: Iterable[int]) -> Table:
        table_selections = {}
        for t in range(len(self.__tables)):
            table_selections[t] = []
        new_assignments = FrozenList()
        for s in selected:
            old_assignment = self.__table_assignments[s]
            table = old_assignment.table
            new_assignments.append(TableAssignment(table=table, column=len(table_selections[table])))
            table_selections[table].append(old_assignment.column)
        new_assignments.freeze()
        new_tables = []
        for t in range(len(self.__tables)):
            tab_sel = table_selections[t]
            new_tables.append(self.__tables[t].select_cols(selected=tab_sel))
        return CollapsedTables(tables=new_tables, table_assignments=new_assignments)

    def colnames(self) -> Sequence[str]:
        table_colnames = [t.colnames() for t in self.__tables]
        return [table_colnames[ta.table][ta.column] for ta in self.__table_assignments]

    def rownames(self) -> Sequence[str]:
        if len(self.__tables) == 0:
            return []
        else:
            return self.__tables[0].rownames()

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


class FreshCollapsedTables(Table):
    """Collapsed tables but without the table assignments (column filtering). They are faster."""
    __tables: Sequence[Table]
    __assign_cache: Optional[FrozenList[TableAssignment]]

    def __init__(self, tables: Sequence[Table], assign_cache: Optional[FrozenList[TableAssignment]] = None,
                 check_rownames: bool = False):
        """Pass RenamedTables to have different name schemas for the different tables."""
        self.__assign_cache = assign_cache
        n_tables = len(tables)
        if n_tables == 0:
            self.__tables = []
        else:
            self.__tables = list(tables)
            if n_tables > 1:
                nrows = self.__tables[0].n_row()
                for i in range(1, len(self.__tables)):
                    if self.__tables[i].n_row() != nrows:
                        raise ValueError("All tables must have the same number of rows.")
                if check_rownames:
                    rownames = self.__tables[0].rownames()
                    for i in range(1, len(self.__tables)):
                        if not equal_iterables(rownames, self.__tables[i].rownames()):
                            raise ValueError("All tables must have the same row names.")

    @staticmethod
    def create_renamed(tables: Sequence[Table]) -> Table:
        if len(tables) == 1:
            return RenamedColsTable(
                table=tables[0], renaming_function=PrefixRenamingFunction(prefix=view_prefix(view_index=0)))
        else:
            return FreshCollapsedTables(
                tables=[RenamedColsTable(table=tables[i],
                                         renaming_function=PrefixRenamingFunction(
                                         prefix=view_prefix(view_index=i))) for i in range(len(tables))])

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

    def to_numpy(self) -> ndarray:
        if len(self.__tables) == 0:
            return np.zeros(shape=(0, 0))
        else:
            return np.concatenate([t.to_numpy() for t in self.__tables], axis=1)

    def inner_to_dataframe(self) -> DataFrame:
        if len(self.__tables) == 0:
            return DataFrame(np.zeros(shape=(0, 0)))
        else:
            res = pandas.concat(objs=[t.to_dataframe() for t in self.__tables], axis=1, copy=True)
            return res

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            from util.table.backed_table import BackedTable
            from util.table.table_backend.np_table import NpTable
            return BackedTable(
                backend=NpTable(data=self.to_numpy(), rownames=self.rownames(), colnames=self.colnames()))

    def memory_size(self) -> int:
        res = 0
        for t in self.__tables:
            res += t.memory_size()
        return res

    def select_rows(self, selected: Iterable[int]) -> Table:
        res_tables = []
        for t in self.__tables:
            res_tables.append(t.select_rows(selected=selected))
        return FreshCollapsedTables(tables=res_tables, assign_cache=self.__assign_cache)

    def __init_assign_cache(self):
        assign = FrozenList(
            (TableAssignment(table=t, column=i)
             for t in range(len(self.__tables))
             for i in range(self.__tables[t].n_col())))
        assign.freeze()
        self.__assign_cache = assign

    def select_cols(self, selected: Iterable[int]) -> Table:
        """There can be a faster version for masks. Additionally, we could avoid to create an assign cache if
        the selected columns are few."""
        if self.__assign_cache is None:
            self.__init_assign_cache()
        table_selections = {}
        for t in range(len(self.__tables)):
            table_selections[t] = []
        new_assignments = FrozenList()
        for s in selected:
            old_assignment = self.__assign_cache[s]
            table = old_assignment.table
            new_assignments.append(TableAssignment(table=table, column=len(table_selections[table])))
            table_selections[table].append(old_assignment.column)
        new_assignments.freeze()
        new_tables = [self.__tables[t].select_cols(selected=table_selections[t]) for t in range(len(self.__tables))]
        return CollapsedTables(tables=new_tables, table_assignments=new_assignments)

    def colnames(self) -> Sequence[str]:
        res = []
        for t in self.__tables:
            res.extend(t.colnames())
        return res

    def rownames(self) -> Sequence[str]:
        if len(self.__tables) == 0:
            return []
        else:
            return self.__tables[0].rownames()

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
