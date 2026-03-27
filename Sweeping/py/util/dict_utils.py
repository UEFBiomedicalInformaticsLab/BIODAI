from util.math.summer import KahanSummer
from typing import Iterable, Any, Sequence, TypeVar, Mapping
from collections import defaultdict


def dict_select(old_dict: dict, keys: Iterable) -> dict:
    """Raises exception if a key is not present."""
    return {k: old_dict[k] for k in keys}


def sorted_dict(d: dict) -> dict:
    """Sorted by keys."""
    res = {}
    for key, value in sorted(d.items(), key=lambda x: x[0]):
        res[key] = value
    return res


def dict_sort_by_value(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1]))


def mean_of_dicts(dicts: Sequence[dict[Any, float]]) -> dict[Any, float]:
    """Keys not present in a dict are considered zero for the mean."""
    to_average = {}
    for d in dicts:
        for k in d:
            if k in to_average:
                to_average[k].append(d[k])
            else:
                to_average[k] = [d[k]]
    res = {}
    num_elems = float(len(dicts))
    for k in to_average:
        res[k] = KahanSummer.sum(to_average[k]) / num_elems
    return res


def validate_dict_values(
        d: dict[Any, set],
        allowed: Iterable,
        check_values: bool = False,
        check_key_in_values: bool = False
) -> bool:
    """
    Validate that all keys and all elements in the sets are in the allowed sequence.
    Optionally check that no key is contained in its own value set.

    Args:
        d: Dictionary to validate.
        allowed: Sequence of allowed strings.
        check_values: if True checks if the values are in the allowed set.
        check_key_in_values (bool): If True, return False if any key appears in its own value set.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    allowed_set = set(allowed)

    # Check keys
    for key, values in d.items():
        if key not in allowed_set:
            return False

        if check_values:
            # Check values are allowed
            if not values.issubset(allowed_set):
                return False

        # Extra check: key should not appear in its own values
        if check_key_in_values and key in values:
            return False

    return True


def key_in_values(parsed_dict: dict[Any, set]) -> bool:
    """
    Return True if any key appears in its own value set.
    """
    for key, values in parsed_dict.items():
        if key in values:
            return True
    return False


def prune_empty_sets(data: dict[Any, set]) -> dict[Any, set]:
    """
    Return a new dictionary with keys removed if their value is an empty set.

    Args:
        data (dict[Any, set]): Input dictionary.

    Returns:
        dict[Any, set]: New dictionary without keys that map to empty sets.
    """
    return {k: v for k, v in data.items() if v}  # `if v` checks for non-empty set


def merge_common_keys(dicts: Iterable[dict]) -> dict[Any, Sequence]:
    """
    Merge multiple dictionaries into a single dictionary where each key maps to
    a sequence of all values associated with that key across the input dictionaries.

    Args:
        dicts (Iterable[dict]): An iterable of dictionaries to merge.

    Returns:
        dict[Any, Sequence]: A dictionary where:
            - Each key is any key found in the input dictionaries.
            - Each value is a list of all values corresponding to that key in the input dictionaries,
              in the order they appear.

    Key insertion order:
        - Keys in the resulting dictionary are inserted in the order they are first encountered
          while iterating through the input dictionaries.
        - For example, if the first dictionary contains keys {"a", "b"} and the second contains {"b", "c"},
          the resulting dictionary will have keys in the order: ["a", "b", "c"].

    Example:
        dicts = [{"a": 1, "b": 2}, {"a": 3, "c": 4}, {"b": 5, "c": 6}]
        merge_common_keys(dicts)
        {'a': [1, 3], 'b': [2, 5], 'c': [4, 6]}
    """
    merged = defaultdict(list)  # Creates a key-list item when accessing a non-existing key.
    for d in dicts:
        for key, value in d.items():
            merged[key].append(value)
    return dict(merged)  # This way the returned object behaves like a basic dict.


def keys_are_sorted(d: dict) -> bool:
    keys = list(d.keys())
    return keys == sorted(keys)


T = TypeVar("T")

def unique_sorted_values(d: Mapping[object, T]) -> list[T]:
    """Return a sorted list of unique values in the dict."""
    return sorted(set(d.values()))


def nested_unique_sorted_values(d: Mapping[object, Iterable[T]]) -> list[T]:
    """Return a sorted list of unique elements from the Iterable values in the dict."""
    res_set = set()
    for _, value in d.items():
        res_set.update(value)
    return sorted(res_set)


def sorted_values_set(d: dict) -> set:
    """
    Returns a set of all values in the dictionary, sorted in ascending order.

    Parameters:
        d (dict): The input dictionary.

    Returns:
        set: A set containing the sorted values.
    """
    return set(sorted(d.values()))


def subset_dict_comprehension(d: dict, keys: Iterable) -> dict:
    """
    Return a subset of the dictionary containing only the specified keys.

    Parameters:
        d (dict): The original dictionary.
        keys (Iterable): Iterable of keys to include in the subset.

    Returns:
        dict: A new dictionary with the selected keys and their values.

    Raises:
        KeyError: If any key in 'keys' is not present in 'd'.
    """
    return {k: d[k] for k in keys}  # Will raise KeyError if missing
