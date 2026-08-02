# Training

Model training lives outside the application package: it needs the full
torch/ultralytics stack, runs on a workstation or GPU box, and must not be
importable from the web tier.

## Layout

```
training/
├── dataset.yaml               class list + split paths (the model's contract)
├── train.py                   training entry point
├── prepare_dataset.py         filename normalisation
├── datasets/                  your images and labels (gitignored)
├── runs/                      training output (gitignored)
└── reference/                 prior runs and sample radiographs
```

## Setup

```bash
pip install -e ".[ml]"
```

Arrange the dataset in YOLO layout:

```
training/datasets/panoramic/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Point `dataset.yaml` at it, then normalise filenames if needed:

```bash
python training/prepare_dataset.py --root training/datasets/panoramic          # dry run
python training/prepare_dataset.py --root training/datasets/panoramic --apply
```

The dry run is the default: renaming in place can silently overwrite a file
whose target name already exists, orphaning its label.

## Training

```bash
python training/train.py --epochs 100 --device mps      # Apple Silicon
python training/train.py --epochs 100 --device 0        # CUDA
```

Before starting, the script checks `dataset.yaml`'s class list against
`dentist_ai.ml.taxonomy` and refuses to run on a mismatch. A reordered class
list does not crash anything — it produces a model whose class 7 means
something different from what the product says class 7 means, which is the
worst failure mode available to a clinical tool. Override with
`--no-verify-classes` only if you know why.

Augmentation is constrained for radiographs: horizontal and vertical flips are
disabled, because a mirrored panoramic swaps left and right quadrants and
teaches anatomy that does not exist. Hue and saturation jitter are off too —
the input is greyscale.

Runs resume automatically from `runs/<name>/weights/last.pt`.

## Deploying a model

```bash
cp training/runs/dentist-ai/weights/best.pt models/best.pt
```

Then set `DENTIST_AI__ML__BACKEND=yolo` in `.env` and restart.

Weights are gitignored. They are large, opaque to code review, and committing
them makes every future clone download every past revision. Distribute them
through a release artefact, object storage, or Git LFS.

## Changing the class list

Appending a class means editing **both** `dataset.yaml` and
`dentist_ai.ml.taxonomy.FINDING_CLASSES`, keeping indices aligned. Append
only — never reorder or delete, because `findings.class_id` rows already
persisted refer to the old positions. (`findings.class_key` is stored
alongside precisely so historical rows survive a taxonomy edit.)
