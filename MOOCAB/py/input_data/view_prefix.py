from collections.abc import Sequence

PREFIX_CONNECTOR = "_"


def create_view_prefix(view_index: int) -> str:
    return str(view_index) + PREFIX_CONNECTOR


def add_view_prefix(name: str, view_num: int) -> str:
    return create_view_prefix(view_index=view_num) + name


def remove_view_prefix(name: str) -> (str, int):
    parts = name.split(PREFIX_CONNECTOR, 1)
    view_num = int(parts[0])
    return parts[1], view_num


def view_number(feature_name: str) -> int:
    return remove_view_prefix(name=feature_name)[1]


def n_views(names: Sequence[str]) -> int:
    name_set = set()
    for n in names:
        name_set.add(view_number(feature_name=n))
    return len(name_set)


def all_unprefixed(names: Sequence[str]) -> Sequence[str]:
    return [(remove_view_prefix(n))[0] for n in names]
