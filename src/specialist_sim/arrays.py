"""Array dtypes as names, so a signature says what it takes."""

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]
UInt32Array = npt.NDArray[np.uint32]
