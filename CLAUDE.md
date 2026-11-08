# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phases 1–5 (training, shared preprocessing, FastAPI service, face detection with `unknown`, tests and load measurements) are done. `?detect=false` skips face detection so LFW-format inputs keep the exact accuracy check against the manifest. `docs/PLAN.md` is the phased plan (in Korean) and `reference/original_train.py` is the original LFW coursework script.

## Commands

```
uv sync                       # install from uv.lock into .venv
uv run python -m ml.train     # train both models, write artifacts/
uv run python -m ml.calibrate # add face-crop config and unknown threshold to manifests; rerun after ml.train, which overwrites them and the service needs them
uv run python -m ml.evaluate  # reload models, compare test accuracy with manifests
uv run python -m ml.mismatch  # accuracy under deliberately wrong preprocessing
uv run uvicorn app.main:app --no-access-log   # needs artifacts/, or ARTIFACTS_DIR
uv run pytest tests           # single test: add -k test_golden_matches_fetch_lfw_people
uv run python scripts/load_test.py   # p50/p95 latency and server memory
```

Change dependencies with `uv add` / `uv remove`, or edit `pyproject.toml` and run `uv lock`; commit `uv.lock`.

Python 3.14 needs the pinned `tensorflow==2.22.0rc0` and `keras-nightly`: stable TensorFlow is not offered to pip on 3.14, so `pip install tensorflow` fails with "No matching distribution".

`docs/PLAN.md` is the source of truth for scope, layout, API contract, and error codes. Work one phase at a time, with one commit per phase. The minimum finished scope is Phases 1–3 and 5.

## What is being built

A FastAPI service that serves two 62-person LFW face classifiers, a CNN (`.keras`) and PCA+KNN (a single `sklearn.Pipeline` saved with joblib), later deployed with Docker on AWS ECS. `ml/` holds training and preprocessing, `app/` holds serving, and `artifacts/<model>-v1/` holds the git-ignored models, which `ml/train.py` regenerates.

## Core invariant: training/serving preprocessing parity

Training and serving must both use `ml/preprocess.py`, and the fitted PCA and preprocessing settings are saved and reused, never recomputed. Each model's `manifest.json` records its labels, input size, preprocessing parameters, test accuracy, library versions, and seed.

- Serving must reproduce `fetch_lfw_people(resize=0.7)`: crop funneled images with `slice_=(70:195, 78:172)`, then resize to 87×65.
- The verified order is RGB crop, BICUBIC resize, divide by 255, then an unweighted channel mean, so don't use PIL's `convert("L")`. Details and the mismatch experiment are in `docs/PREPROCESS.md`.
- A golden test compares `preprocess.py` output on raw LFW JPGs with the `fetch_lfw_people` arrays.
- The script also rescales pixels by 255 only if the max exceeds 1.0 (`reference/original_train.py:105`). Serving has to end up in the same value range as training.

## Porting `reference/original_train.py` to `ml/train.py`

The script is reference material: don't import it from new code. Drop the visualization and keep data handling identical so results stay comparable:

- `RANDOM_SEED = 0`, `set_random_seed`, and `enable_op_determinism`
- `min_faces_per_person=20`, all 62 classes, 20 images per class drawn with `np.random.RandomState(0)`
- Stratified 75/25 train/test split, then a stratified 15% CNN validation split taken from train
- CNN architecture as defined in the script, 40 epochs, batch size 32

Two intentional changes from the script:

- Select the PCA+KNN configuration by validation or CV within train. The script picks it by test accuracy; test is for final evaluation only.
- Restrict KNN to k≥3, because k=1 yields only 0/1 probabilities and makes the top-3 response meaningless.

Phase 1 is done when reloaded models match the post-training test accuracy.

## Serving constraints

- Load models once in the lifespan handler.
- Declare inference endpoints with `def`, not `async def`, because inference is blocking CPU work.
- Enforce file size and pixel limits before decoding, and apply EXIF rotation.
- Never store or log uploaded images. Real people's faces are involved, so the ALB also stays restricted to the owner's IP rather than being a public demo.

## Repository publishing

Coursework rules on publishing the original code to a public repo must be confirmed first (`docs/PLAN.md`). Don't bring in files with a student ID in the name, PDF reports, `.venv`, or `figures`.
