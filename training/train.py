#!/usr/bin/env python3
"""Train the detection model.

Separate from the application package: training needs the full
torch/ultralytics stack, runs on a workstation or a GPU box, and has no
business being importable from the web tier.

    python training/train.py --epochs 100 --device mps

Every setting is an argument, so a run is reproducible from its command line.
Resume points at the directory Ultralytics actually writes to. And the
dataset's class list is verified against the application's taxonomy before a
run starts, so reordered classes fail in seconds instead of producing weights
that mislabel every finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
DEFAULT_DATA = TRAINING_ROOT / "dataset.yaml"
DEFAULT_RUNS = TRAINING_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", default="yolov8m.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu | mps | 0 | 0,1 — 'mps' uses Apple Silicon acceleration",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="dentist-ai")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--no-verify-classes",
        action="store_true",
        help="Skip the taxonomy consistency check (not recommended)",
    )
    return parser.parse_args()


def verify_classes(data_path: Path) -> None:
    """Fail fast if the dataset's classes drifted from the application's.

    A mismatch here does not crash anything — it produces a model whose class
    7 means something different from what the product says class 7 means,
    which is the worst possible failure mode for a clinical tool.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from dentist_ai.ml.taxonomy import FINDING_CLASSES  # noqa: PLC0415

    with data_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_names: dict[int, str] = config.get("names", {})
    expected = {item.class_id: item.label("ru") for item in FINDING_CLASSES}

    if dataset_names != expected:
        missing = set(expected) - set(dataset_names)
        extra = set(dataset_names) - set(expected)
        mismatched = {
            index: (dataset_names[index], expected[index])
            for index in set(dataset_names) & set(expected)
            if dataset_names[index] != expected[index]
        }
        print("Class list does not match dentist_ai.ml.taxonomy:", file=sys.stderr)
        if missing:
            print(f"  missing from dataset: {sorted(missing)}", file=sys.stderr)
        if extra:
            print(f"  unknown in dataset:   {sorted(extra)}", file=sys.stderr)
        for index, (found, want) in sorted(mismatched.items()):
            print(f"  [{index}] dataset={found!r} taxonomy={want!r}", file=sys.stderr)
        raise SystemExit(1)

    print(f"✓ {len(expected)} classes match the application taxonomy")


def main() -> None:
    args = parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}")

    if not args.no_verify_classes:
        verify_classes(args.data)

    from ultralytics import YOLO  # noqa: PLC0415 - heavy import, script-only

    # Resume from the checkpoint Ultralytics actually writes, not a guessed path.
    checkpoint = args.project / args.name / "weights" / "last.pt"
    resume = checkpoint.is_file()

    if resume:
        print(f"↻ Resuming from {checkpoint}")
        model = YOLO(str(checkpoint))
    else:
        print(f"▶ Starting fresh from {args.weights}")
        model = YOLO(args.weights)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=resume,
        # Radiographs are greyscale and orientation-critical: a flipped
        # panoramic swaps left and right quadrants, which would teach the model
        # anatomy that does not exist.
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.08,
        scale=0.25,
        patience=25,
    )

    print(f"\n✓ Training complete. Weights and metrics: {results.save_dir}")
    print(f"  Deploy with: cp {results.save_dir}/weights/best.pt models/best.pt")


if __name__ == "__main__":
    main()
