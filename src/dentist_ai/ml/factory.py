"""Detector composition root."""

from __future__ import annotations

from dentist_ai.core.config import Settings
from dentist_ai.core.logging import get_logger
from dentist_ai.ml.detector import Detector
from dentist_ai.ml.stub import StubDetector

log = get_logger(__name__)


def build_detector(settings: Settings) -> Detector:
    if settings.ml.backend == "stub":
        log.info("detector_selected", backend="stub")
        return StubDetector()

    # Imported lazily: this module is on the import path of every process,
    # including CI, where torch is not installed.
    from dentist_ai.ml.yolo import YoloDetector  # noqa: PLC0415 - deferred torch import

    log.info(
        "detector_selected",
        backend="yolo",
        weights=str(settings.ml.resolved_weights_path),
        device=settings.ml.device,
    )
    return YoloDetector(
        settings.ml.resolved_weights_path,
        device=settings.ml.device,
        confidence_threshold=settings.ml.confidence_threshold,
        iou_threshold=settings.ml.iou_threshold,
        max_detections=settings.ml.max_detections,
        worker_threads=settings.ml.worker_threads,
    )
