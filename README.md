# TraceME Pipeline

TraceME is a video tracking pipeline that runs SAM2 on frame sequences, produces per-frame CSV summaries, and exports annotated videos.

## What You Get
- Annotated MP4 outputs per chunk (and a merged MP4).
- Per-frame CSVs per chunk (and a merged CSV).
- A CLI for running the pipeline over frame folders and prompt YAMLs.

## Requirements
- Python 3.12+
- SAM2 (installed automatically via pip)
- CUDA GPU recommended for speed (CPU is supported but slow)

## Install (PyPI)
```
pip install traceme-pipeline
```

### Checkpoints (Auto-Download)
TraceME automatically downloads the required SAM2 checkpoint on first run if it is missing.
By default, checkpoints are stored in:
```
~/.cache/traceme/sam2/checkpoints
```

You can also pre-download all checkpoints:
```
traceme-download-checkpoints
```

To change the checkpoint location:
- `SAM2_CHECKPOINT_DIR=/path/to/checkpoints`

To disable auto-download:
- `TRACEME_AUTO_DOWNLOAD=0`

### Dev Helper (SAM2 + Checkpoints)
If you prefer a local clone of SAM2 (for development), you can still use:
```
make install
```
This creates a venv, installs TraceME in editable mode, clones SAM2 into `third_party/sam2`, and downloads checkpoints.

### Configure SAM2 (optional)
If SAM2 lives somewhere else, set:
- `SAM2_ROOT=/path/to/sam2` (should contain `sam2/` and `checkpoints/`)
- `SAM2_MODEL=tiny|small|base_plus|large|sam3` (default: `large`)

### SAM3 (optional)
The pipeline can also run Meta's SAM3 tracker with the same points/boxes prompt
workflow:
```
pip install 'traceme-pipeline[sam3]'   # requires Python >= 3.12
traceme -i frames/ -o out/ -p prompts.yaml --model sam3
```
The `sam3.pt` checkpoint is resolved through the same logic as the SAM2
checkpoints (`SAM2_CHECKPOINT`, `SAM2_CHECKPOINT_DIR`, or the default cache
dir), but it is **not** auto-downloaded: it is gated on Hugging Face, so log
in with `hf auth login`, download it from the `facebook/sam3` repo, and place
it in your checkpoint directory as `sam3.pt`.

## Usage
Run the pipeline:
```
traceme -i /path/to/frames -o /path/to/output -p /path/to/prompts.yaml
```

Generate a tasks file for batch runs:
```
traceme-gen-tasks /path/to/root -o tasks.tsv
```

## Prompt YAML Format
Prompt files must contain a top-level `prompts` list. See `src/traceme/prompts/parser.py` for the exact schema and examples.

## Outputs
Given `frame_dir=/data/frames/clipA`, outputs are:
- `/output/clipA.csv` (merged CSV)
- `/output/clipA.mp4` (merged annotated video; one color and `id:` label per tracked object; exactly one video frame per input frame, so video frame N corresponds to `global_frame_idx` N)
- `/output/clipA_run_summary.json` (run status, processed/resumed/failed chunk ids, totals)
- `/output/clipA_tmp/` (intermediate chunk files; removed if `--del_tmp` is set). Chunk folders contain symlinks to the original frames (falling back to copies on filesystems without symlink support), so chunking costs almost no disk space.

CSV columns: `chunk_id, global_frame_idx, in_chunk_idx, obj_id, area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h`. Frames with no tracked objects produce a single row with an empty `obj_id` and `area_px=0`. If an object is tracked but the model loses its mask for a given frame (empty prediction), every stat column for that row is `-1`.

## Saving Masks (for shape analysis)
Pass `--save-masks` to persist every object's binary mask, bit-packed, into
`/output/clipA_masks.npz` — one merged archive per video (per-chunk masks are
written alongside each chunk's CSV/video during the run, then combined the
same way the merged CSV/video are, so `--del_tmp` does **not** remove them).
Only non-empty masks are stored.

Reload the mask archive:
```python
import numpy as np
from traceme.sam2.io import _unpack_mask

data = np.load("clipA_masks.npz", allow_pickle=True)
for gidx, oid, packed, shp in zip(
    data["global_frame_idx"], data["obj_id"], data["packed"], data["shape"]
):
    mask = _unpack_mask(packed, tuple(shp))  # bool array, shape (H, W)
    # e.g. shape descriptors via skimage:
    # from skimage.measure import regionprops, label
    # props = regionprops(label(mask))[0]
    # props.eccentricity, props.perimeter, props.solidity, ...
```

## Resume & Failures
- Completed chunks are marked in the tmp folder; re-running the same command skips them and continues from the first incomplete chunk. Use `--no-resume` to reprocess everything.
- If any chunk fails, the pipeline still merges what it has, marks the run `"partial"` in the run summary, keeps the tmp folder (even with `--del_tmp`), and exits with code 1. Re-run the same command to retry only the failed chunks.

## Troubleshooting
- If SAM2 configs or checkpoints are missing, TraceME will raise a clear error with the expected paths.
- Set `HYDRA_FULL_ERROR=1` for detailed SAM2 errors.

## License
See `LICENSE`.
