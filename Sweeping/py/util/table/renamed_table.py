from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Iterable, Sequence

from frozenlist import FrozenList
from numpy import ndarray
from pandas import DataFrame

from util.table.backed_table import BackedTable
from util.table.table import Table
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_backend.np_table import NpTable


class RenamingFunction(ABC):

    @abstractmethod
    def apply(self, position: int, original: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def select(self, selected: Optional[Iterable[int]]) -> RenamingFunction:
        raise NotImplementedError()

    @abstractmethod
    def is_valid(self, size: int) -> bool:
        """Checks if this renaming function is valid for the given number of names."""
        raise NotImplementedError()


class PrefixRenamingFunction(RenamingFunction):
    __prefix: str

    def __init__(self, prefix: str):
        self.__prefix = prefix

    def apply(self, position: int, original: str) -> str:
        return self.__prefix + original

    def select(self, selected: Optional[Iterable[int]]) -> RenamingFunction:
        return self

    def is_valid(self, size: int) -> bool:
        return True


class OverwriteRenamingFunction(RenamingFunction):
    __new_names: FrozenList[str]

    def __init__(self, new_names: Iterable[str]):
        self.__new_names = FrozenList(new_names)
        self.__new_names.freeze()

    def apply(self, position: int, original: str) -> str:
        try:
            return self.__new_names[position]
        except IndexError as ie:
            raise IndexError(str(ie)+"\n"+
                             "index: " + str(position) + "\n" +
                             "original: " + str(original) + "\n")

    def select(self, selected: Optional[Iterable[int]]) -> RenamingFunction:
        if selected is None:
            return self
        else:
            names = self.__new_names
            return OverwriteRenamingFunction(new_names=(names[s] for s in selected))

    def is_valid(self, size: int) -> bool:
        return size == len(self.__new_names)


class RenamedTable(Table, ABC):
    """A table where either rows or columns are renamed."""
    __inner: Table
    __renaming_function: RenamingFunction

    def __init__(self, table: Table, renaming_function: RenamingFunction):
        self.__inner = table
        self.__renaming_function = renaming_function

    def _inner(self) -> Table:
        return self.__inner

    def _renaming(self) -> RenamingFunction:
        return self.__renaming_function

    def n_row(self) -> int:
        return self.__inner.n_row()

    def n_col(self) -> int:
        return self.__inner.n_col()

    def to_numpy(self) -> ndarray:
        return self.__inner.to_numpy()

    def compile(self, max_cells: Optional[int] = DEFAULT_MAX_CACHEABLE_CELLS) -> Table:
        if max_cells is not None and self.size() > max_cells:
            return self
        else:
            return BackedTable(
                backend=NpTable(data=self.to_numpy(), rownames=self.rownames(), colnames=self.colnames()))

    def memory_size(self) -> int:
        return self.__inner.memory_size()

    def has_fast_cols(self) -> bool:
        return self.__inner.has_fast_cols()

    def has_fast_rows(self) -> bool:
        return self.__inner.has_fast_rows()


class RenamedColsTable(RenamedTable):
    """A table where columns are renamed."""

    def __init__(self, table: Table, renaming_function: RenamingFunction):
        RenamedTable.__init__(self=self, table=table, renaming_function=renaming_function)
        if not renaming_function.is_valid(size=table.n_col()):
            raise ValueError("Renaming function not valid for the number of cols.")

    def select_rows(self, selected: Iterable[int]) -> Table:
        return RenamedColsTable(
            table=self._inner().select_rows(selected=selected), renaming_function=self._renaming())

    def select_cols(self, selected: Iterable[int]) -> Table:
        return RenamedColsTable(
            table=self._inner().select_cols(selected=selected),
            renaming_function=self._renaming().select(selected=selected))

    def inner_to_dataframe(self) -> DataFrame:
        df = self._inner().inner_to_dataframe()
        func = self._renaming()
        df.columns = [func.apply(position=i, original=c) for i, c in enumerate(df.columns)]
        return df

    def colnames(self) -> Sequence[str]:
        func = self._renaming()
        return [func.apply(position=i, original=c) for i, c in enumerate(self._inner().colnames())]

    def rownames(self) -> Sequence[str]:
        return self._inner().rownames()


class RenamedRowsTable(RenamedTable):
    """A table where rows are renamed."""

    def __init__(self, table: Table, renaming_function: RenamingFunction):
        RenamedTable.__init__(self=self, table=table, renaming_function=renaming_function)
        if not renaming_function.is_valid(size=table.n_row()):
            raise ValueError("Renaming function not valid for the number of rows.")

    def select_rows(self, selected: Iterable[int]) -> Table:
        return RenamedRowsTable(
            table=self._inner().select_rows(selected=selected),
            renaming_function=self._renaming().select(selected=selected))

    def select_cols(self, selected: Iterable[int]) -> Table:
        return RenamedRowsTable(
            table=self._inner().select_cols(selected=selected), renaming_function=self._renaming())

    def inner_to_dataframe(self) -> DataFrame:
        df = self._inner().inner_to_dataframe()
        func = self._renaming()
        df.index = [func.apply(position=i, original=c) for i, c in enumerate(df.index)]
        return df

    def colnames(self) -> Sequence[str]:
        return self._inner().colnames()

    def rownames(self) -> Sequence[str]:
        func = self._renaming()
        return [func.apply(position=i, original=c) for i, c in enumerate(self._inner().rownames())]
