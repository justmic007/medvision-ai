#!/usr/bin/env bash
# Fetch public sample chest X-rays for local development and demo.
#
# These are public, de-identified test images from the TorchXRayVision repo.
# Per DECISIONS.md D-03, image data is NEVER committed to this repo — this
# script reproduces the local data/ folder on any clone instead.
#
# Usage:  bash scripts/fetch_sample_data.sh
set -euo pipefail

BASE="https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests"
mkdir -p data

echo "Fetching sample chest X-rays into data/ ..."

# Primary demo image (clean, well-separated findings — good for showing capability).
curl -fL -o data/sample_cxr.jpg     "$BASE/16747_3_1.jpg"

# NIH ChestX-ray14 samples (in-distribution for the model).
curl -fL -o data/nih_01.png         "$BASE/00000001_000.png"
curl -fL -o data/nih_02.png         "$BASE/00027426_000.png"

# COVID pneumonia case (a known-finding image — useful for discussing limits).
curl -fL -o data/covid_pneumonia.jpg "$BASE/covid-19-pneumonia-58-prior.jpg"

echo "Done. Fetched:"
ls -1 data/*.jpg data/*.png 2>/dev/null
