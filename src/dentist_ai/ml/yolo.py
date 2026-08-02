"""Ultralytics YOLO backend.

``model.predict`` is a multi-second CPU-bound call and the model object is not
thread-safe, so inference runs in a bounded thread pool with a semaphore that
both caps concurrency and serialises access to the model.

Imports of ``ultralytics`` are deferred to ``load()``: a deployment running
the stub backend never pays the multi-second torch import.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image

from dentist_ai.core.errors import InferenceError
from dentist_ai.core.logging import get_logger
from dentist_ai.ml.detector import Detection, InferenceResult

if TYPE_CHECKING:
    from ultralytics import YOLO

log = get_logger(__name__)

_WARM_UP_SIZE: Final[tuple[int, int]] = (640, 640)


class YoloDetector:
    """Ultralytics detector with bounded, off-loop execution."""

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        worker_threads: int = 2,
    ) -> None:
        self._weights_path = weights_path
        self._device = device
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._max_detections = max_detections
        self._model: YOLO | None = None
        self._executor = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="yolo")
        self._semaphore = asyncio.Semaphore(worker_threads)
        self._load_lock = asyncio.Lock()
        self._version = f"yolo:{weights_path.stem}"

    @property
    def model_version(self) -> str:
        return self._version

    async def warm_up(self) -> None:
        """Load weights and run one throwaway pass.

        The first inference is several times slower than steady state because
        of lazy kernel initialisation. Paying it at startup keeps the first
        real user off the slow path.
        """
        model = await self._ensure_loaded()
        blank = Image.new("RGB", _WARM_UP_SIZE, color=(0, 0, 0))
        loop = asyncio.get_running_loop()
        async with self._semaphore:
            await loop.run_in_executor(self._executor, self._infer, model, blank)
        log.info("detector_warmed_up", version=self._version, device=self._device)

    async def detect(self, image: Image.Image) -> InferenceResult:
        model = await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        async with self._semaphore:
            raw = await loop.run_in_executor(self._executor, self._infer, model, image)
        duration_ms = max(1, int((time.perf_counter() - started) * 1000))
        return InferenceResult(
            detections=raw,
            model_version=self._version,
            duration_ms=duration_ms,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- internals --------------------------------------------------------
    async def _ensure_loaded(self) -> YOLO:
        # Double-checked locking: the fast path avoids the lock entirely once
        # the model is resident, while concurrent first requests load it once.
        model = self._model
        if model is not None:
            return model
        async with self._load_lock:
            cached = self._model
            if cached is not None:  # another coroutine won the race
                return cached
            loaded = await asyncio.to_thread(self._load)
            self._model = loaded
            return loaded

    def _load(self) -> YOLO:
        if not self._weights_path.is_file():
            log.error("weights_missing", path=str(self._weights_path))
            raise InferenceError("Файл весов модели не найден. Проверьте ML__WEIGHTS_PATH.")
        try:
            import ultralytics  # noqa: PLC0415 - optional dependency, loaded on demand
        except ImportError as exc:
            log.error("ultralytics_missing")
            raise InferenceError(
                'Пакет ultralytics не установлен. Выполните: pip install -e ".[ml]"'
            ) from exc

        model = ultralytics.YOLO(str(self._weights_path))
        log.info("model_loaded", path=str(self._weights_path), classes=len(model.names))
        return model

    def _infer(self, model: YOLO, image: Image.Image) -> tuple[Detection, ...]:
        """Run the model. Executed on a worker thread — never on the loop."""
        frame = np.asarray(image.convert("RGB"))
        height, width = frame.shape[:2]

        results = model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou,
            max_det=self._max_detections,
            device=self._device,
            verbose=False,
        )
        if not results:
            return ()

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()

        # One vectorised transfer off the accelerator instead of a per-box
        # `.cpu()` round-trip per detection in a Python loop.
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(confidence),
                    x=float(x1) / width,
                    y=float(y1) / height,
                    width=float(x2 - x1) / width,
                    height=float(y2 - y1) / height,
                ).clamped()
            )
        return tuple(detections)
