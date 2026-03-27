from collections.abc import Sequence


class FallbackLocations:
    __locations: dict[str, Sequence[str]]

    def __init__(self, locations: dict[str, Sequence[str]]):
        self.__locations = locations

    def locations_for_view(self, view_name: str) -> Sequence[str]:
        return self.__locations.get(view_name, ())


EMPTY_FALLBACK_LOCATIONS = FallbackLocations(locations=dict())