from __future__ import annotations

import numpy as np

DEFAULT_MAX_CACHEABLE_CELLS = 30000000  # Was 20000000, but was not enough for TCGA-BRCA
DEFAULT_CHUNK_CELLS = DEFAULT_MAX_CACHEABLE_CELLS
"""Was 20000000 before setting it equal to DEFAULT_MAX_CACHEABLE_CELLS.
100000000 can be faster but depletes memory on some architectures."""
TABLE_DTYPE = np.number
