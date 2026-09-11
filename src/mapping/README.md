# Starting-lineup mapping

This package maps text visibly present in a football lineup graphic. It does not invent or externally canonicalize player identities.

Pipeline:

1. Scout the detected lineup interval at 2 FPS with PP-OCR Tiny.
2. Detect multiple semantic panels: starter pitch, starter list, substitutes, and coach.
3. Exclude substitutes/coach/UI and group temporally stable starter graphics.
4. Pick the best candidate frame plus nearby candidate frames (3 initially, 7 on fallback).
5. Re-run PP-OCR Small, pair names and numbers globally one-to-one, then apply multi-frame consensus.
6. Write complete 11-player lineups to `resolved_starters.csv`; retain partial/conflicting results only in diagnostics and evidence.

From an existing lineup detection JSON:

```bash
python run_graphics.py \
  --detection-json outputs/Tv360/TV360_8/detection_result.json \
  --output-dir outputs/mapping_v3/TV360_8
```

For one known interval:

```bash
python run_graphics.py data/match.mp4 --start 94 --end 120 \
  --output-dir outputs/mapping_v3/match
```
