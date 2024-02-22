import itertools
from collections.abc import Sequence, Iterable
from typing import Any

import pandas as pd
from pandas import DataFrame

from input_data.view_prefix import PREFIX_CONNECTOR
from util.dataframes import prefix_all_cols, nan_slice, select_by_row_indices
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.list_like import ListLike


def collapse_views(views, verbose=False) -> DataFrame:
    views_keys = list(views.keys())
    n_views = len(views_keys)
    if n_views == 0:
        return pd.DataFrame()
    if verbose:
        print("Collapsing views")
        print("Number of views: " + str(n_views))
        for k in views_keys:
            print("View " + str(k) + " number of columns: " + str(len(views[k].columns)))
            print("View " + str(k) + " duplicated columns: " +
                  str(views[k].loc[:, views[k].columns.duplicated()].columns))
    res = prefix_all_cols(views[views_keys[0]], "0" + PREFIX_CONNECTOR)  # Assuming there is at least a view
    for i in range(1, n_views):
        k = views_keys[i]
        res = pd.concat([res, prefix_all_cols(views[k], str(i) + PREFIX_CONNECTOR)], axis=1)
    if verbose:
        print("Duplicated columns after collapse: " + str(res.loc[:, res.columns.duplicated()].columns))
    return res


def collapse_feature_importance(distributions: Sequence[Distribution]) -> Distribution:
    """Each distribution is multiplied by its len / total len."""
    tot_len = 0
    for d in distributions:
        tot_len += len(d)
    res_list = []
    if tot_len > 0:
        for d in distributions:
            mult = len(d) / tot_len
            res_list.extend([i * mult for i in d])
    return ConcreteDistribution(probs=res_list)


def filter_by_mask(x, mask: ListLike) -> DataFrame:
    """Mask is applied on columns."""
    n_cols = len(x.columns)
    len_mask = len(mask)
    if n_cols != len_mask:
        raise ValueError(
            "Number of columns and length of mask differ.\n" +
            "Number of columns: " + str(n_cols) + "\n" +
            "mask length: " + str(len_mask) + "\n"
            "mask as list: " + str(list(mask)) + "\n")
    positions = mask.true_positions()
    res = x.iloc[:, positions]
    return res


def collapse_views_and_filter_by_mask(views, mask: ListLike) -> DataFrame:
    collapsed = collapse_views(views)
    return filter_by_mask(collapsed, mask)


def collapse_iterable_dfs_and_filter_by_mask(dfs: Iterable[DataFrame], mask: ListLike) -> DataFrame:
    dic = {}
    i = 0
    for v in dfs:
        dic[str(i)] = v
    collapsed = collapse_views(dic)
    return filter_by_mask(collapsed, mask)


def mv_select_by_indices(views, indices) -> dict[str, Any]:
    """Selects rows."""
    res = {}
    for v in views:
        res[v] = views[v].iloc[indices]
    return res


def mv_select_all_sets(views, y, fold):
    """ Selects all sets of samples for the passed fold. """
    train_indices = fold[0]
    test_indices = fold[1]
    x_train = mv_select_by_indices(views, train_indices)
    y_train = select_by_row_indices(y, train_indices)
    x_test = mv_select_by_indices(views, test_indices)
    y_test = select_by_row_indices(y, test_indices)
    return x_train, y_train, x_test, y_test


def check_masks(views, masks):
    collapsed = collapse_views(views)
    mask = itertools.chain.from_iterable(masks)
    filtered = filter_by_mask(collapsed, mask)
    nan_s = nan_slice(filtered)
    if not nan_s.empty:
        raise ValueError("Check masks failed\n" +
                         "NaN slice:\n" +
                         str(nan_s))
