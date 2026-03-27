import json

from typing import Sequence, Iterable, Optional

from util.named import Named
from util.dict_utils import sorted_dict


def pretty_duration(duration_seconds) -> str:
    d = int(duration_seconds)
    days = d // 86400
    hours = d % 86400 // 3600
    mins = d % 3600 // 60
    secs = d % 60
    return '{:02d}-{:02d}:{:02d}:{:02d}'.format(days, hours, mins, secs)


def name_str(x):
    if isinstance(x, Named):
        return x.name()
    else:
        return str(x)


def name_value(name: str, value) -> str:
    if isinstance(name, Named):
        str_name = name.name()
    else:
        str_name = str(name)
    if isinstance(value, Named):
        str_val = value.name()
    elif isinstance(value, str):
        str_val = value
    elif isinstance(value, Sequence):
        str_val = names_str(value)
    else:
        str_val = str(value)
    return str_name + " = " + str_val


def feature_names(column_names: list[str], active_feature_positions: set[int]) -> list[str]:
    res = []
    for a in active_feature_positions:
        res.append(column_names[a])
    return res


def feature_names_from_collapsed_views(collapsed_views, active_feature_positions: set[int]) -> list[str]:
    return feature_names(column_names=collapsed_views.columns, active_feature_positions=active_feature_positions)


def automatic_to_string(x) -> str:
    return str(vars(x))


def str_dict(d: dict, in_lines: bool = False) -> str:
    res = ""
    if not in_lines:
        res += "{"
    first = True
    for key in d:
        value = d[key]
        if first:
            first = False
        else:
            if in_lines:
                res += "\n"
            else:
                res += ", "
        res += str(key) + ": " + str(value)
    if not in_lines:
        res += "}"
    return res


def str_sorted_dict(d: dict, in_lines: bool = False) -> str:
    return str_dict(sorted_dict(d), in_lines=in_lines)


def str_paste(parts: Iterable[str], separator: str = ", ") -> str:
    return iterable_to_string(li=parts, separator=separator, compact=True, brackets=False, max_len=None)


def str_in_lines(li: Iterable) -> str:
    res = ""
    for i in li:
        res += str(i) + "\n"
    return res


def tuple_to_string(tup: Iterable, compact: bool = False, max_len: Optional[int] = 100) -> str:
    return iterable_to_string(li=tup, compact=compact, max_len=max_len, bracket_type=("(",")"))


def iterable_to_string(li: Iterable, compact=False, separator=",", brackets: bool = True,
                       max_len: Optional[int] = 100,
                       bracket_type: tuple[str,str] = ("[","]")) -> str:
    """If not compact there will be a space in addition to the separator.
    Every element is converted to string if needed."""
    res = ""
    if brackets:
        res += bracket_type[0]
    for i,e in enumerate(li):
        if i > 0:
            if compact:
                res += separator
            else:
                res += separator + " "
        if max_len is None or i < max_len:
            res += str(e)
        else:
            res += "..."
            break
    if brackets:
        res += bracket_type[1]
    return res


def names(it: Iterable[Named]) -> list[str]:
    return [i.name() for i in it]


def elem_names(elems: Sequence) -> Sequence[str]:
    strings = []
    for e in elems:
        strings.append(name_str(e))
    return strings


def names_str(elems: Sequence) -> str:
    return iterable_to_string(elem_names(elems))


def proportion_str(proportion: float) -> str:
    return str(round(proportion, 2))


def fdr_str(fdr_threshold: float) -> str:
    return proportion_str(proportion=fdr_threshold)


def parse_json_dict_property(value: str, allow_list_as_dict: bool = False) -> dict[str, set[str]]:
    """
    Safely parse a JSON-like property string into a Python dictionary
    with string keys and sets of strings as values. Supports values as
    lists or sets in the input JSON. Keys and values are normalized to
    alphabetical order, and all strings are stripped of whitespace.
    Accepts strings of this kind: {"mrna": ["age"]}
    Or of this kind: ["mrna", "age"]

    Args:
        value (str): The property value containing JSON-like content.
        allow_list_as_dict (bool): If True, interpret a top-level list as dict keys with empty values.

    Returns:
        dict[str, set[str]]: Parsed dictionary with sorted keys and sorted sets.

    Raises:
        ValueError: If the string is not valid JSON or structure is incorrect.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e.msg} at position {e.pos}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error while parsing JSON: {str(e)}") from e

    # Handle case where parsed is a list
    if isinstance(parsed, list):
        if allow_list_as_dict:
            if not all(isinstance(item, str) for item in parsed):
                raise ValueError("Invalid list: expected list of strings.")
            # Convert list elements to keys with empty sets
            return {item.strip(): set() for item in sorted(parsed)}

    # Handle case where parsed is a dict
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not a dictionary or list.\n" +
                         "Parsed JSON:\n" +
                         str(parsed))

    # Normalize: sort keys and sort values inside sets
    result: dict[str, set[str]] = {}
    for k in sorted(parsed.keys()):
        key = k.strip()
        v = parsed[k]
        if isinstance(v, (list, set)):
            if not all(isinstance(item, str) for item in v):
                raise ValueError(f"Invalid value for key '{k}': expected list/set of strings.")
            stripped_values = [item.strip() for item in v]
            result[key] = set(sorted(stripped_values))
        else:
            raise ValueError(f"Invalid value for key '{k}': expected list or set of strings.")

    return result


def has_duplicates_case_insensitive(strings: Iterable[str]) -> bool:
    normalized = [s.lower() for s in strings]
    return len(normalized) != len(set(normalized))
