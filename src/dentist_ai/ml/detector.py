"""Detector abstraction.

Inference sits behind a ``Protocol`` so the test suite runs without torch,
nothing above this module knows what a tensor is, and a model that fails to
load degrades to a 503 instead of crashing the process at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detection with box coordinates normalised to ``[0, 1]``."""

    class_id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def clamped(self) -> Detection:
        """Clip to the unit square.

        Detectors routinely emit boxes a pixel or two outside the frame; the
        database has a CHECK constraint that would reject them, so normalise
        here rather than discovering it at INSERT time.
        """
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        return Detection(
            class_id=self.class_id,
            confidence=min(max(self.confidence, 0.0), 1.0),
            x=x,
            y=y,
            width=min(max(self.width, 1e-6), 1.0 - x),
            height=min(max(self.height, 1e-6), 1.0 - y),
        )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    model_version: str
    duration_ms: int


@runtime_checkable
class Detector(Protocol):
    """Anything that can turn an image into detections."""

    @property
    def model_version(self) -> str: ...

    async def warm_up(self) -> None:
        """Optional: pay one-time initialisation cost before serving traffic."""

    async def detect(self, image: Image.Image) -> InferenceResult: ...
