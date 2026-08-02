"""Deterministic fake detector.

The default backend, so a fresh clone runs end to end without a 2 GB torch
install. Output is seeded from the image content: the same radiograph always
yields the same findings.
"""

from __future__ import annotations

import hashlib
import random
import time

from PIL import Image

from dentist_ai.ml.detector import Detection, InferenceResult
from dentist_ai.ml.taxonomy import FINDING_CLASSES, Category

_PLAUSIBLE_CLASSES = tuple(
    item.class_id
    for item in FINDING_CLASSES
    if item.category in (Category.PATHOLOGY, Category.RESTORATION, Category.CONDITION)
)


class StubDetector:
    """Content-seeded pseudo-detections."""

    def __init__(self, *, min_findings: int = 3, max_findings: int = 9) -> None:
        self._min = min_findings
        self._max = max_findings

    @property
    def model_version(self) -> str:
        return "stub-1.0.0"

    async def warm_up(self) -> None:
        return None

    async def detect(self, image: Image.Image) -> InferenceResult:
        started = time.perf_counter()
        rng = random.Random(self._seed(image))  # noqa: S311 - not security-sensitive

        count = rng.randint(self._min, self._max)
        detections: list[Detection] = []
        for _ in range(count):
            width = rng.uniform(0.04, 0.13)
            height = rng.uniform(0.05, 0.16)
            detections.append(
                Detection(
                    class_id=rng.choice(_PLAUSIBLE_CLASSES),
                    confidence=round(rng.uniform(0.28, 0.97), 4),
                    x=rng.uniform(0.02, 1.0 - width - 0.02),
                    y=rng.uniform(0.05, 1.0 - height - 0.05),
                    width=width,
                    height=height,
                ).clamped()
            )

        duration_ms = max(1, int((time.perf_counter() - started) * 1000))
        return InferenceResult(
            detections=tuple(detections),
            model_version=self.model_version,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _seed(image: Image.Image) -> int:
        thumb = image.convert("L").resize((32, 32))
        digest = hashlib.sha256(thumb.tobytes()).digest()
        return int.from_bytes(digest[:8], "big")
