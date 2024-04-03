from __future__ import annotations

from enum import Enum


class OutcomeType(Enum):
    categorical = 1
    survival = 2
    other = 3
