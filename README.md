# TraceWave Pipeline

TraceWave is a video tracking pipeline that runs SAM2 on frame sequences, produces per-frame CSV summaries, and exports annotated videos.

## What You Get
- Annotated MP4 outputs per chunk (and a merged MP4).
- Per-frame CSVs per chunk (and a merged CSV).
- A CLI for running the pipeline over frame folders and prompt YAMLs.

## Requirements
- Python 3.10+
- SAM2 (installed automatically via pip)
- CUDA GPU recommended for speed (CPU is supported but slow)

## Install (PyPI)
```
pip install tracewave
```

### Checkpoints (Auto-Download)
TraceWave automatically downloads the required SAM2 checkpoint on first run if it is missing.
By default, checkpoints are stored in:
```
~/.cache/tracewave/sam2/checkpoints
```

You can also pre-download all checkpoints:
```
tracewave-download-checkpoints
```

To change the checkpoint location:
- `SAM2_CHECKPOINT_DIR=/path/to/checkpoints`

To disable auto-download:
- `TRACEWAVE_AUTO_DOWNLOAD=0`

### Dev Helper (SAM2 + Checkpoints)
If you prefer a local clone of SAM2 (for development), you can still use:
```
make install
```
This creates a venv, installs TraceWave in editable mode, clones SAM2 into `third_party/sam2`, and downloads checkpoints.

### Configure SAM2 (optional)
If SAM2 lives somewhere else, set:
- `SAM2_ROOT=/path/to/sam2` (should contain `sam2/` and `checkpoints/`)
- `SAM2_MODEL=tiny|small|base_plus|large` (default: `large`)

## Usage
Run the pipeline:
```
tracewave -i /path/to/frames -o /path/to/output -p /path/to/prompts.yaml
```

Generate a tasks file for batch runs:
```
tracewave-gen-tasks /path/to/root -o tasks.tsv
```

## Prompt YAML Format
Prompt files must contain a top-level `prompts` list. See `src/tracewave/prompts/parser.py` for the exact schema and examples.

## Outputs
Given `frame_dir=/data/frames/clipA`, outputs are:
- `/output/clipA.csv` (merged CSV)
- `/output/clipA.mp4` (merged annotated video)
- `/output/clipA_tmp/` (intermediate chunk files; removed if `--del_tmp` is set)

## Troubleshooting
- If SAM2 configs or checkpoints are missing, TraceWave will raise a clear error with the expected paths.
- Set `HYDRA_FULL_ERROR=1` for detailed SAM2 errors.

## License
See `LICENSE`.
