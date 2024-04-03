from typing import Sequence

from location_manager.location_manager import LocationManager
from setup.evaluation_setup import DEFAULT_SEED


class BasicLocationManager(LocationManager):
    """This legacy location manager ignores the seed."""

    def _seed_adder(self, before_seed_path: str, seed: int) -> str:
        return before_seed_path

    def _seeds_to_check(self, before_seed_path: str) -> Sequence[int]:
        return [DEFAULT_SEED]
