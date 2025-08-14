grid-cropper
============

Utility to extract individual business cards from a scanned multi-pocket “cardholder” sheet. It detects or enforces a grid, crops each cell, optionally tightens to the actual card edges, and saves outputs as `ROW-COL.png`.

Features
--------
- Grid detection from horizontal/vertical line masks or uniform split.
- Optional deskew (Hough-based) before detection.
- Tighten each cell to actual card edges (robust to dashed lines and gaps).
- Row-aware alignment and normalization:
  - Normalize card heights across the sheet.
  - Align crops per row by top/bottom edge.
  - Normalize widths per row while respecting individual cell bounds.
- Deterministic file naming with optional zero-padding.
- Debug overlays and intermediate images.

Environment
-----------
If you have a venv:
```bash
source /Users/rayriffy/Downloads/nei/.venv/bin/activate.fish
```

Basic usage
-----------
```bash
python3 /Users/rayriffy/Downloads/nei/extract_cards.py \
  --input /Users/rayriffy/Downloads/nei/01-1.png \
  --output /Users/rayriffy/Downloads/nei/01-1 \
  --rows 2 --cols 5 \
  --include-image-borders \
  --zero-pad \
  --tighten-to-card --tighten-size-constrained \
  --first-row-at-image-top --anchor-top-max-shift-px 12 \
  --card-aspect 1.75 --card-aspect-tolerance 0.15 \
  --normalize-height --height-tolerance-ratio 0.04 \
  --row-align auto --row-align-tolerance-px 4 \
  --normalize-width-per-row --width-tolerance-ratio 0.05 \
  --debug
```

If detection misfires, fall back to a uniform grid with the same rows/cols:
```bash
python3 /Users/rayriffy/Downloads/nei/extract_cards.py \
  --input /Users/rayriffy/Downloads/nei/01-1.png \
  --output /Users/rayriffy/Downloads/nei/01-1 \
  --rows 2 --cols 5 \
  --force-equal-grid \
  --zero-pad \
  --tighten-to-card --tighten-size-constrained \
  --first-row-at-image-top --anchor-top-max-shift-px 12 \
  --card-aspect 1.75 --card-aspect-tolerance 0.15 \
  --normalize-height --height-tolerance-ratio 0.04 \
  --row-align auto --row-align-tolerance-px 4 \
  --normalize-width-per-row --width-tolerance-ratio 0.05 \
  --debug
```

Processing pipeline
-------------------
1. Load image, convert to gray.
2. Adaptive threshold to binary; optional deskew via Hough lines if skew > ~0.5°.
3. Morphological extraction of horizontal/vertical line masks.
4. Reduce masks to 1D line positions and (optionally) add image borders.
5. Fit to target grid counts with 1D k-means or uniform fallback; or enforce equal grid.
6. For each cell, apply inner margin, then optionally tighten to card edges:
   - Combines Otsu and Canny with morphology to bridge dashed lines and remove grid residues.
   - Enforces aspect and area sanity; optional top-edge anchoring for first row.
7. Post passes on final rectangles before saving:
   - Normalize height based on dominant height cluster.
   - Align vertically per row (auto/top/bottom).
   - Normalize width per row to dominant width (bounded by that cell’s limits).

Key flags
--------
- Grid control:
  - `--rows N`, `--cols M`: expected row/column counts.
  - `--force-equal-grid`: skip detection, split uniformly using rows/cols.
  - `--include-image-borders`: treat image edges as grid lines.
- Detection params:
  - `--length-threshold f` (default 0.5): min fraction of orthogonal dimension to accept a line.
  - `--inner-margin-px n` (default 5): shave inside each grid cell to avoid line artifacts.
- Tightening to card:
  - `--tighten-margin-px n` (default 2): inset when tightening.
  - `--card-aspect a` (default 1.75), `--card-aspect-tolerance t` (default 0.15).
  - `--first-row-at-image-top`: prefer top-anchored card in first row.
  - `--anchor-top-max-shift-px n` (default 12): allowable slack from the cell’s top.
- Height normalization and row alignment:
  - `--normalize-height`: use the dominant height across all cards.
  - `--height-tolerance-ratio r` (default 0.04): tolerance for height clustering.
  - `--row-align {auto,top,bottom}` (default auto): align cards per row by the straighter edge.
  - `--row-align-tolerance-px n` (default 4): only shift if deviation exceeds this.
- Per-row width normalization:
  - `--normalize-width`: use dominant width within each row.
  - `--width-tolerance-ratio r` (default 0.05): tolerance for width clustering per row.

Tuning notes
------------
- Line residue in crops: increase `--inner-margin-px` to 8–12.
- Plastic/pocket edges leaking in: increase `--tighten-margin-px` to 4–6.
- Too few lines detected: lower `--length-threshold` to 0.35–0.45 and/or `--include-image-borders`.
- If cards vary slightly in size: keep `--normalize-height` on; width differences handled by `--normalize-width` per row.

Outputs
-------
- Cropped cards as `ROW-COL.png` saved to the output directory.
- Debug (when `--debug`):
  - `_debug_01_binary.png`: adaptive threshold result.
  - `_debug_02_horizontal_mask.png`: horizontal line mask.
  - `_debug_03_vertical_mask.png`: vertical line mask.
  - `_debug_04_crop_boxes.png`: final rectangles actually saved (after tighten/normalize/align).

Troubleshooting
---------------
- Detection unstable: try `--force-equal-grid` with known `--rows/--cols`.
- First-row cards not hugging the top: ensure `--first-row-at-image-top` and adjust `--anchor-top-max-shift-px`.
- Misaligned within a row: keep `--row-align auto` or force `--row-align top|bottom`.

```
python3 /Users/rayriffy/Downloads/nei/extract_cards.py \
        --input /Users/rayriffy/Downloads/nei/sample_01-2.png \
        --output /Users/rayriffy/Downloads/nei/01-2 \
        --rows 2 --cols 5 \
        --include-image-borders \
        --first-row-at-image-top --anchor-top-max-shift-px 12 \
        --card-aspect 1.75 --card-aspect-tolerance 0.15 \
        --row-align auto --row-align-tolerance-px 4 \
        --normalize-height --height-tolerance-ratio 0.04 \
        --normalize-width --width-tolerance-ratio 0.05
```