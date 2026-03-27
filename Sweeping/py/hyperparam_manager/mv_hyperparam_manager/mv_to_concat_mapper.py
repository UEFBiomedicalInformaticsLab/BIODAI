from util.list_like import BoolListLike, ListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet


class MvToConcatMapper:
    """Maps from single view masks to a concatenated mask and vice versa."""
    __view_sizes: dict[str, int]
    __concat_size: int

    def __init__(self, view_sizes: dict[str, int]):
        """View names are sorted alphabetically during construction."""
        self.__view_sizes = {s: view_sizes[s] for s in sorted(view_sizes.keys())}
        self.__concat_size = sum(self.__view_sizes.values())

    def __init_mv_masks(self) -> dict[str, BoolListLike]:
        return {name: SparseBoolListBySet(min_size=size) for name, size in self.__view_sizes.items()}

    def concat_to_mv_masks(self, concat_mask: ListLike) -> dict[str, BoolListLike]:
        """Views with 0 active features are still returned."""
        if len(concat_mask) != self.__concat_size:
            raise ValueError(
                "Size mismatch. Passed mask: " + str(len(concat_mask)) +
                " Expected: " + str(self.__concat_size) + "\n" +
                "Self: " + str(self) + "\n")
        res = self.__init_mv_masks()
        items = iter(self.__view_sizes.items())
        current_item = next(items)
        view_start = 0
        view_name = current_item[0]
        view_next_start = view_start + current_item[1]
        for pos in concat_mask.true_positions():
            while pos >= view_next_start:
                current_item = next(items)
                view_start = view_next_start
                view_name = current_item[0]
                view_next_start = view_start + current_item[1]
            res[view_name][pos-view_start]=True
        return res

    def mt_to_concat_mask(self, mv_masks: dict[str, ListLike]) -> BoolListLike:
        view_sizes = self.__view_sizes
        local_keys = view_sizes.keys()
        if mv_masks.keys() != local_keys:
            raise ValueError()
        res = SparseBoolListBySet()
        view_start = 0
        for k in local_keys:
            v = mv_masks[k]
            view_size = view_sizes[k]
            if len(v) != view_size:
                raise ValueError()
            for view_pos in v.true_positions():
                res.set_true(key=view_start+view_pos)
            view_start = view_start + view_size
        assert len(res) == self.__concat_size
        return res

    def concat_size(self) -> int:
        return self.__concat_size

    def __str__(self) -> str:
        return str(self.__view_sizes)

    def view_size(self, view_name: str) -> int:
        return self.__view_sizes[view_name]
