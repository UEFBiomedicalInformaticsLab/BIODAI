from __future__ import annotations
from collections import Sequence

from pandas import DataFrame

from util.distribution.distribution import Distribution
from util.list_like import ListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet


class Mask:
    __bool_mask: ListLike[bool]

    def __init__(self, mask: Sequence[bool]):
        self.__bool_mask = SparseBoolListBySet(seq=mask)

    def mask_positions(self) -> Sequence[int]:
        return self.__bool_mask.true_positions()

    def mask_len(self) -> int:
        return len(self.__bool_mask)

    def apply_df(self, df: DataFrame) -> DataFrame:
        """Mask is applied on columns."""
        mask = self.__bool_mask
        n_cols = len(df.columns)
        len_mask = len(mask)
        if n_cols != len_mask:
            raise ValueError(
                "Number of columns and length of mask differ.\n" +
                "Number of columns: " + str(n_cols) + "\n" +
                "mask length: " + str(len_mask) + "\n"
                "mask as list: " + str(list(mask)) + "\n")
        return df.iloc[:, mask.true_positions()]

    def apply_backward_seq(self, seq: Sequence[float]) -> Sequence[float]:
        seq_len = len(seq)
        mask_pos = self.mask_positions()
        n_mask_pos = len(mask_pos)
        if seq_len != n_mask_pos:
            raise ValueError(
                "Passed sequence length and active positions in mask differ.\n" +
                "Passed sequence length: " + str(seq_len) + "\n" +
                "Active positions in mask: " + str(n_mask_pos) + "\n")
        res = [0.0]*self.mask_len()
        for i, x in zip(mask_pos, seq):
            res[i] = x
        return res

    @staticmethod
    def from_distribution(d: Distribution) -> Mask:
        return Mask(d.nonzero())
