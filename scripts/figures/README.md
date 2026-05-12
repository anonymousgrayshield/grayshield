# Camera-Ready RQ Figure Scripts

These standalone plotting scripts regenerate the compact RQ figures used by
the paper from JSONL result artifacts.

The recommended entry point is:

```bash
python scripts/generate_camera_ready_figures.py
```

The wrapper creates a temporary `datasets/` directory from bundled
`release_results/rq*/rq*.jsonl` files, executes the scripts in this directory,
and copies the generated PNG/PDF files to
`release_results/camera_ready_figures/`.

Direct execution is also possible if the current working directory contains:

```text
datasets/rq1.jsonl
datasets/rq2.jsonl
datasets/rq3.jsonl
```

