"""Array dtypes as names, so a signature says what it takes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
