"""Array dtypes as names, so a signature says what it takes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]
