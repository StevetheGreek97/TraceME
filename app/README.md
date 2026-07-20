# TraceWave

TraceWave is a desktop app for annotating video frames with points, boxes, and polygons, with SAM2-assisted segmentation. It turns videos into frame sequences, organizes work into projects, autosaves your state, and exports per-video YAML prompts for downstream pipelines.

**Highlights**
- Project-based workflow with autosave and resume state.
- Video import and frame extraction via FFmpeg.
- Fast frame navigation with a frame list, slider, and keyboard shortcuts.
- Multi-object annotations using object IDs.
- Point, box, and polygon mask editing.
- Optional SAM2-assisted polygon generation.
- YAML export per video plus a project-wide JSON store.

**Requirements**
- Python 3.9 or newer.
- PyQt6 and NumPy.
- FFmpeg and FFprobe available on your PATH for video import.
- Optional for SAM2: PyTorch, OpenCV, and the `sam2` package.
- Optional for YAML export: PyYAML.

**Quick Install (macOS/Linux)**
The included installer sets up a virtual environment, installs all extras, and downloads the SAM2 weights.

```bash
./install.sh
./run.sh
```

**Quick Install (Windows)**
Use the Windows batch scripts from CMD or PowerShell:

```bat
install.bat
run.bat
```

Alternatively, you can run the Bash scripts using Git Bash or WSL:

```bash
./install.sh
./run.sh
```

**Manual Install**
If you want full control, you can install manually.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[sam2,yaml]"
```

Download SAM2 weights:

```bash
make sam2-weights
```

Launch:

```bash
python -m src.tracewave
```

**Conda Install (Windows/macOS/Linux)**
If you prefer Anaconda/Miniconda, use a dedicated env and install with pip.
Note: the `sam2` package is not available on conda channels, so you must use pip
inside the conda env (or omit the `sam2` extra).

```bash
conda create -n tracewave python=3.10 -y
conda activate tracewave
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip
python -m pip install -e ".[sam2,yaml]"
```

If you do not want SAM2, install only core + YAML:

```bash
python -m pip install -e ".[yaml]"
```

Download SAM2 weights (only if you installed the `sam2` extra):

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt','src/sam2_configs/sam2_hiera_tiny.pt')"
```

Run:

```bash
python -m src.tracewave
```

**What Gets Installed**
- Core: `PyQt6`, `numpy`
- Extras: `torch`, `opencv-python`, `sam2`, `pyyaml`
- `sam2` is installed via pip (not conda).
- SAM2 weights are downloaded to `src/sam2_configs/sam2_hiera_tiny.pt`

**Run Options**
- `./run.sh` uses `.venv/bin/python` by default, or an active conda env if present.
- Windows: `run.bat` uses `.venv\Scripts\python.exe`.
- You can override with `PYTHON=/path/to/python ./run.sh`.
- You can also run directly with `python -m src.tracewave`.

**Using the App**
1. File > New Project (or Open Project).
2. File > Import Videos to extract frames.
3. Select a video from the project tree.
4. Annotate frames in the main view.
5. File > Export YAMLs for per-video prompt files.

**Controls**
- Left click: add a positive point.
- Right click: add a negative point.
- Ctrl+click a point: remove it.
- Shift+drag: draw a bounding box.
- Arrow keys: previous/next frame.
- Home/End: first/last frame.
- E: run SAM2 on the current object.
- D: clear the current object's polygon mask.

**Project Layout**
A project is a folder containing a `project.json`, an `annotations.json`, and a frames directory.

```text
my_project/
  project.json
  annotations.json
  frames/
    my_video/
      00000.jpg
      00001.jpg
```

**Data Formats**
- `annotations.json` stores all annotations across videos, keyed by `video_id`, `frame_idx`, and `obj_id`.
- Exported YAML contains a `prompts` list with `frame_idx`, `obj_id`, `points`, `labels`, and optional `box` or `polygon` fields.

**SAM2 Configuration**
TraceWave reads SAM2 settings from each project’s `project.json` under the `sam2` key.

Fields:
- `config_name`: SAM2 config name or path.
- `weights_path`: absolute path or path relative to the repo root.

Default weights path:
- `src/sam2_configs/sam2_hiera_tiny.pt`

If you move the weights file, update `weights_path` in your project’s `project.json`. The app also attempts common fallbacks so older paths keep working.

**Make Targets**
```bash
make venv
make install
make install-full
make sam2-weights
make run
make clean
```

**Scripts**
- `install.sh` always installs full extras and downloads SAM2 weights.
- `run.sh` starts the app using the virtual environment.

**Troubleshooting**
- “FFmpeg not found”: install FFmpeg and ensure `ffmpeg` and `ffprobe` are on your PATH.
- “SAM2 dependencies are not available”: install `torch` and `sam2`.
- “OpenCV is required for SAM2 mask conversion”: install `opencv-python`.
- “PyYAML not installed”: install `pyyaml`.
- “No such file or directory ... sam2_hiera_tiny.pt”: run `./install.sh` or `make sam2-weights`.

**Notes on Large Files**
The SAM2 weights are not committed to GitHub. The repo’s `.gitignore` excludes `src/sam2_configs/` so weights stay local.

**License**
MIT. See `LICENSE`.
