from collections.abc import Iterable, Sequence

from pandas import DataFrame

from util.dataframes import select_cols_by_mask
from util.sequence_utils import filter_by_booleans, flatten_iterable_of_iterable


class FeatureSpaceLifter:
    __active_features: Sequence[bool]  # Features in bigger space that are in smaller space.

    def __init__(self, active_features: Sequence[bool]):
        self.__active_features = active_features

    def uplift(self, features: Iterable[bool]) -> list[bool]:
        """Features from bigger space to smaller space."""
        return filter_by_booleans(features, self.__active_features)

    def downlift(self, features: Sequence[bool]) -> list[bool]:
        """Features from smaller space to bigger space."""
        if not isinstance(features, Sequence):
            raise ValueError("features not sequence. Features: " + str(features))
        af = self.__active_features
        big_len = len(af)
        res = [False]*big_len
        j = 0
        for i in range(big_len):
            if af[i]:
                try:
                    res[i] = features[j]
                except IndexError as e:
                    raise IndexError(
                        "i: " + str(i) + "\n" +
                        "j: " + str(j) + "\n" +
                        "self features len: " + str(big_len) + "\n" +
                        "features type: " + str(type(features)) + "\n" +
                        "features: " + str(features) + "\n" +
                        "original error: " + str(e) + "\n")
                j += 1
        return res

    def uplift_df(self, df: DataFrame) -> DataFrame:
        """Columns from bigger space to smaller space."""
        return select_cols_by_mask(df=df, mask=self.__active_features)

    def active_features_mask(self) -> Sequence[bool]:
        return self.__active_features

    def __str__(self) -> str:
        return str(self.__active_features)


class FeatureSpaceLifterMV:
    __single_view_lifters: Iterable[FeatureSpaceLifter]

    def __init__(self, single_view_lifters: Iterable[FeatureSpaceLifter]):
        self.__single_view_lifters = single_view_lifters

    def uplift(self, features_mv: Iterable[Iterable[bool]]) -> list[list[bool]]:
        """Features from bigger space to smaller space."""
        return [lifter.uplift(features=f) for lifter, f in zip(self.__single_view_lifters, features_mv)]

    def downlift(self, features_mv: Iterable[Sequence[bool]]) -> list[list[bool]]:
        """Features from smaller space to bigger space."""
        return [lifter.downlift(features=f) for lifter, f in zip(self.__single_view_lifters, features_mv)]

    def uplift_dfs(self, dfs: Iterable[DataFrame]) -> list[DataFrame]:
        """Columns from bigger space to smaller space."""
        return [lifter.uplift_df(df=df) for lifter, df in zip(self.__single_view_lifters, dfs)]

    def collapse(self) -> FeatureSpaceLifter:
        return FeatureSpaceLifter(
            active_features=flatten_iterable_of_iterable(
                x=[lifter.active_features_mask() for lifter in self.__single_view_lifters]))

    def uplift_dict(self, views: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Uplifts a dictionary [name, dataframe]"""
        res = {}
        for v, lift in zip(views.keys(), self.__single_view_lifters):
            res[v] = lift.uplift_df(views[v])
        return res
