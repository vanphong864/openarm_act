"""Which decoder reads the LeRobot dataset's videos: torchcodec or PyAV.

torchcodec needs the system's FFmpeg libraries (libavutil.so.5x). LeRobot's
own check only imports the package, which succeeds without them, and the
first frame then fails with "Could not load libtorchcodec". This loads the
decoders, and falls back to PyAV, which ships its own FFmpeg: slower, same
frames. Installing FFmpeg (`sudo apt install ffmpeg`) keeps torchcodec.
"""

from __future__ import annotations

import functools
import logging

log = logging.getLogger(__name__)


@functools.cache
def video_backend() -> str:
    try:
        import torchcodec.decoders  # noqa: F401  -- loads libtorchcodec and FFmpeg
    except (ImportError, OSError, RuntimeError):
        log.warning("torchcodec cannot load FFmpeg here; decoding videos with pyav (apt install ffmpeg to use torchcodec)")
        return "pyav"
    return "torchcodec"
