# PlantID100 - Lightweight Plant ID for modern mobile apps

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Repo](https://img.shields.io/badge/GitHub-PlantID100-black)](https://github.com/nicholasbc100/PlantID100)

PlantID100 is a compact plant-identification engine designed for **real-world 2026 mobile needs**: quick response, low battery use, and confidence scores that can say "not sure" when needed.

## What it does
- Uses visual embeddings + leaf morphology + color similarity.
- Ranks top species candidates with calibrated confidence.
- Runs in a lightweight way suited to iPhone 7+ and current low-power devices.

## Why this is useful today
- Great fit for offline-first field apps (gardening, education, eco surveys).
- Confidence-aware outputs are safer for user-facing recommendations.
- Easy to extend by adding new `PlantReference` entries without retraining a huge model.

## Quick run
```bash
python3 plant_id.py
```

## Minimal API example
```python
from plant_id import PlantIDEngine, PlantReference

references = [
    PlantReference("acer_rubrum", (0.2, 0.8, 0.1), 1.8, 0.7, (0.18, 0.62, 0.18)),
    PlantReference("ficus_benjamina", (0.7, 0.3, 0.5), 2.4, 0.1, (0.20, 0.52, 0.19)),
]

engine = PlantIDEngine(references)
results = engine.identify(
    image_embedding=(0.25, 0.75, 0.15),
    leaf_aspect_ratio=1.65,
    serration_score=0.62,
    color_profile=(0.20, 0.58, 0.17),
)
```

## Project files
- `plant_id.py`: retrieval + fusion engine and runnable example.
- `tests/test_plant_id.py`: focused regression tests.
