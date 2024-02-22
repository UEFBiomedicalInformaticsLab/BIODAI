from collections.abc import Set


class MVFeatureSetByNames:
    __features_by_view: dict[str, set[str]]

    def __init__(self, features_by_view: dict[str, set[str]]):
        self.__features_by_view = features_by_view

    def view_names(self) -> Set[str]:
        """Returned object is set-like."""
        return self.__features_by_view.keys()

    def view_features(self, view_name: str) -> set[str]:
        return self.__features_by_view[view_name]

    def __str__(self) -> str:
        return str(self.__features_by_view)
