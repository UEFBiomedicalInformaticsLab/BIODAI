from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Iterable, Union, Tuple

from numpy import ndarray
from sympy import nextprime

from util.math.utils import ceil_division
from util.table.table_consts import DEFAULT_MAX_CACHEABLE_CELLS
from util.table.table_utils import n_row, n_col

MAX_STRIPES_TO_CACHE = 2000
STRIPES_PER_CHUNK = 1
"""When changed, datasets should be rebuilt for maximum performance.
Increasing the number of stripes per chunk does not seem to increase performance, not even when multiple adjacent
stripes must be accessed."""
TRANSPOSE_HDF5 = True
"""When chunks are vertical transposing enhances speed for disk operations.
Warning: changing this makes previously written hdf5 files unreadable."""
DEFAULT_BY_COL = not TRANSPOSE_HDF5


def good_indexing(indexing: Optional[Iterable[int]]) -> Optional[Union[list[int], ndarray]]:
    """Transforms to a good indexing for a loaded hdf5 table, if necessary."""
    if indexing is None:
        return indexing
    if isinstance(indexing, (list, ndarray)):
        return indexing
    return list(indexing)


def elements_to_bytes(elements: int) -> int:
    """Assuming float64 are in use."""
    return elements*8


def chunks_to_cache(
        nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL, stripes_per_chunk: int = STRIPES_PER_CHUNK) -> int:
    """If chunks are too big cache is not used at all."""
    return stripes_to_cache(nrow=nrow, ncol=ncol, by_col=by_col) // stripes_per_chunk


def stripes_to_cache(nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL) -> int:
    if by_col:
        return cols_to_cache(nrow=nrow, ncol=ncol)
    else:
        return rows_to_cache(nrow=nrow, ncol=ncol)


def cols_to_cache(nrow: int, ncol: int) -> int:
    return min(
        ncol,
        MAX_STRIPES_TO_CACHE,
        max(DEFAULT_MAX_CACHEABLE_CELLS, 0) // max(1, nrow))


def rows_to_cache(nrow: int, ncol: int) -> int:
    return min(
        nrow,
        MAX_STRIPES_TO_CACHE,
        max(DEFAULT_MAX_CACHEABLE_CELLS, 0) // max(1, ncol))


def n_stripes(nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL) -> int:
    if by_col:
        return ncol
    else:
        return nrow


def n_chunks(num_stripes: int, stripes_per_chunk: int = STRIPES_PER_CHUNK) -> int:
    return ceil_division(num=num_stripes, den=stripes_per_chunk)


def stripe_length(nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL) -> int:
    if by_col:
        return nrow
    else:
        return ncol


def optimal_rdcc_nslots(nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL,
                        stripes_per_chunk: int = STRIPES_PER_CHUNK) -> int:
    chunks_to_c = chunks_to_cache(nrow=nrow, ncol=ncol, by_col=by_col, stripes_per_chunk=stripes_per_chunk)
    if chunks_to_c == 0:
        return 0
    else:
        return min(n_chunks(num_stripes=n_stripes(nrow=nrow, ncol=ncol, by_col=by_col),
                            stripes_per_chunk=stripes_per_chunk),
                   nextprime((chunks_to_c * 100) - 1))


def optimal_rdcc_nbytes(
        nrow: int, ncol: int, by_col: bool = DEFAULT_BY_COL, stripes_per_chunk: int = STRIPES_PER_CHUNK) -> int:
    return elements_to_bytes(
        elements=chunks_to_cache(nrow, ncol, by_col=by_col, stripes_per_chunk=stripes_per_chunk) *
                 stripe_length(nrow=nrow, ncol=ncol, by_col=by_col) *
                 stripes_per_chunk)


class HDF5ChunkStrategy:
    __by_col: bool

    def __init__(self, by_col: bool):
        self.__by_col = by_col

    def optimal_rdcc_nbytes(self, nrow: int, ncol: int) -> int:
        return optimal_rdcc_nbytes(nrow=nrow, ncol=ncol, by_col=self.__by_col, stripes_per_chunk=STRIPES_PER_CHUNK)

    def optimal_rdcc_nslots(self, nrow: int, ncol: int) -> int:
        return optimal_rdcc_nslots(nrow=nrow, ncol=ncol, by_col=self.__by_col, stripes_per_chunk=STRIPES_PER_CHUNK)

    def select(self, data, selected_cols: Sequence[int], selected_rows: Sequence[int]):
        """We have to select in two phases since h5py does not support two indexing vectors.
        We start by selecting stripes for faster access."""
        if self.__by_col:
            return data[:, good_indexing(selected_cols)][selected_rows, :]
        else:
            return data[good_indexing(selected_rows), :][:, selected_cols]

    def chunks(self, data) -> Tuple[int, int]:
        """The shape of a chunk."""
        if self.__by_col:
            return n_row(data), STRIPES_PER_CHUNK
        else:
            return STRIPES_PER_CHUNK, n_col(data)

    def has_fast_cols(self) -> bool:
        return self.__by_col

    def has_fast_rows(self) -> bool:
        return not self.__by_col


DEFAULT_HDF5_CHUNK_STRATEGY = HDF5ChunkStrategy(by_col=DEFAULT_BY_COL)