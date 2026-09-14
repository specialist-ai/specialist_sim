"""A film written frame by frame, H.264 through PyAV."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from specialist_sim.arrays import UInt8Array


@contextmanager
def writer(
  path: Path, width: int, height: int, fps: float
) -> Iterator[Callable[[UInt8Array], None]]:
  """An H.264 film at ``path``; the writer takes each frame, ``(height, width, 3)`` RGB."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with av.open(str(path), "w") as film:
    stream = film.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20"}

    def write(frame: UInt8Array) -> None:
      image = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame, np.uint8), "rgb24")
      film.mux(stream.encode(image))

    yield write
    film.mux(stream.encode())
