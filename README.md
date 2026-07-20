# TraceWave

Monorepo for the TraceWave video tracking toolchain. It has two independent Python projects that share only a YAML prompt file format:

- **[`app/`](app/README.md)** — PyQt6 desktop app for annotating video frames with points, boxes, and polygons, with SAM2-assisted mask suggestions. Exports one `prompts.yaml` per video.
- **[`pipeline/`](pipeline/README.md)** — Headless CLI (published to PyPI as `tracewave`) that takes a frame folder plus an exported `prompts.yaml` and runs SAM2's video predictor to propagate those annotations across every frame, producing annotated MP4s and per-frame area CSVs.

## Workflow

1. Annotate a handful of frames per video in `app/`, export `<video>.yaml`.
2. Run `pipeline/` (via the `tracewave` CLI) against the full frame set and that YAML to propagate the annotations across all frames.

Each project keeps its own `pyproject.toml`, dependencies, and install scripts — see each subdirectory's README for setup instructions. The PyPI release workflow (`.github/workflows/publish.yml`) only builds and publishes `pipeline/` on a manual dispatch or a published GitHub release; it is not triggered by ordinary commits.

## License

See each subproject's `LICENSE` file.
