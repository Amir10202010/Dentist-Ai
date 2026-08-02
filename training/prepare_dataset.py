#!/usr/bin/env python3
"""Normalise dataset filenames to a zero-padded sequence.

    python training/prepare_dataset.py --root training/datasets/panoramic

Renaming files in place with no dry run and no collision
handling: if a target name already existed, ``Path.rename`` silently replaced
it, destroying an image and orphaning its label. This one plans the whole
rename first, refuses to run on a conflict, and stages through temporary names
so a partially-numbered directory can be re-normalised safely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Rename:
    image_from: Path
    image_to: Path
    label_from: Path | None
    label_to: Path | None


def plan_split(images_dir: Path, labels_dir: Path) -> tuple[list[Rename], list[str]]:
    images = sorted(item for item in images_dir.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    renames: list[Rename] = []
    warnings: list[str] = []

    for index, image in enumerate(images, start=1):
        stem = f"{index:05d}"
        label = labels_dir / f"{image.stem}.txt"
        has_label = label.is_file()
        if not has_label:
            warnings.append(f"no label for {image.name}")

        renames.append(
            Rename(
                image_from=image,
                image_to=image.with_name(stem + image.suffix.lower()),
                label_from=label if has_label else None,
                label_to=labels_dir / f"{stem}.txt" if has_label else None,
            )
        )
    return renames, warnings


def apply(renames: list[Rename]) -> None:
    """Two-phase rename via temporary names.

    Renaming directly can collide when the source and target sets overlap
    (re-running on an already-numbered directory). Staging every file to a
    unique temporary name first makes the operation safe to repeat.
    """
    staged: list[tuple[Path, Path]] = []
    for index, item in enumerate(renames):
        temp = item.image_from.with_name(f".staging-{index}{item.image_from.suffix}")
        item.image_from.rename(temp)
        staged.append((temp, item.image_to))

        if item.label_from is not None and item.label_to is not None:
            label_temp = item.label_from.with_name(f".staging-{index}.txt")
            item.label_from.rename(label_temp)
            staged.append((label_temp, item.label_to))

    for temp, final in staged:
        temp.rename(final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Without this the run is a dry run.",
    )
    args = parser.parse_args()

    total = 0
    for split in SPLITS:
        images_dir = args.root / "images" / split
        labels_dir = args.root / "labels" / split

        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"− skipping {split}: missing images/ or labels/")
            continue

        renames, warnings = plan_split(images_dir, labels_dir)
        total += len(renames)
        print(f"\n{split}: {len(renames)} images")
        for warning in warnings[:10]:
            print(f"  ! {warning}")
        if len(warnings) > 10:
            print(f"  ! …and {len(warnings) - 10} more without labels")

        if args.apply:
            apply(renames)
            print(f"  ✓ renamed {len(renames)} images")

    if not args.apply:
        print(f"\nDry run: {total} files would be renamed. Re-run with --apply.")


if __name__ == "__main__":
    main()
