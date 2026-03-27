from typing import Sequence

from numpy import unique

from util.sequence_utils import transpose
from util.str_utils import str_paste


def string_from_selected(parts: Sequence[str], selected: Sequence[bool], separator: str = " ") -> str:
    res = ""
    for p, s in zip(parts, selected):
        if s:
            if res != "" and p != "":
                res += separator
            res += p
    return res


def parts_in_common(object_features: Sequence[Sequence[str]]) -> Sequence[str]:
    """The outer sequence is for objects, the inner sequences are the features.
    Each object must have the same number of features.
    Returns a list of the parts that are in common between all the objects."""
    n_obj = len(object_features)
    if n_obj == 0:
        return []
    transposed = transpose(object_features)
    common = []
    for f in transposed:
        if len(unique(f)) == 1:
            common.append(f[0])
    return common


def group_name(object_features: Sequence[Sequence[str]], separator: str = " ") -> str:
    """The outer sequence is for objects, the inner sequences are the features.
    Each object must have the same number of features.
    Returns a name by pasting the parts that are in common between all the objects."""
    n_obj = len(object_features)
    if n_obj == 0:
        return ""
    transposed = transpose(object_features)
    common = []
    for f in transposed:
        if len(unique(f)) == 1:
            common.append(f[0])
    return str_paste(parts=common, separator=separator)


def names_by_differences(object_features: Sequence[Sequence[str]], separator: str = " ") -> Sequence[str]:
    """The outer sequence is for objects, the inner sequences are the features.
    Each object must have the same number of features.
    Returns a string for each object, avoiding the features that all objects have in common."""
    n_obj = len(object_features)
    if n_obj == 0:
        return []
    transposed = transpose(object_features)
    to_use = []
    for f in transposed:
        if len(unique(f)) > 1:
            to_use.append(True)
        else:
            to_use.append(False)
    return [string_from_selected(parts=o, selected=to_use, separator=separator) for o in object_features]
