# PlantID100 — Faster Plant ID (iPhone 7 friendly)

This is a compact plant-ID approach built to be **fast**, **battery-safe**, and still very accurate on older phones.

## The idea (plain English)
- First, a tiny neural net turns the photo into a short visual fingerprint (embedding).
- Then we mix in cheap leaf-shape signals (like aspect ratio + serration).
- Then we compare against known plant references and rank the best matches.
- Finally we calibrate confidence so the app can say "not sure" instead of guessing wild.

So instead of one giant heavy model, this uses a **hybrid** method that is quicker and usually more stable on hard photos.

## Why this beats a basic classifier
- Better on edge cases: similar-looking species are separated by morphology + color cues.
- Lower latency: tiny encoder + lightweight retrieval.
- More honest confidence: calibration step avoids fake certainty.
- Easy to grow: add references without retraining the whole stack.

## File
- `plant_id.py` contains the retrieval+fusion engine and a runnable example.

## Quick run
```bash
python3 plant_id.py
```
