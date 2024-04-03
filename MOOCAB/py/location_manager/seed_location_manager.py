from collections.abc import Sequence

from location_manager.location_manager import LocationManager


class SeedLocationManager(LocationManager):

    def _seed_adder(self, before_seed_path: str, seed: int) -> str:
        return before_seed_path + str(seed) + "/"

    def _seeds_to_check(self, before_seed_path: str) -> Sequence[int]:
        res = self._seed_directories_from_path(before_seed_path=before_seed_path)
        return res
