#!/usr/bin/env python3
"""Regenerate camera-ready RQ figures from bundled release artifacts.

The plotting scripts in ``scripts/figures`` were authored as standalone
camera-ready scripts that expect ``datasets/rq*.jsonl`` relative to their
working directory.  This wrapper keeps those scripts unchanged while making
them reproducible from the anonymized repository layout:

    release_results/rq1/rq1.jsonl
    release_results/rq2/rq2.jsonl
    release_results/rq3/rq3.jsonl

Generated PNG/PDF artifacts are copied to
``release_results/camera_ready_figures`` by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = ROOT / "scripts" / "figures"

SCRIPTS = [
    "gs_RQ1.py",
    "gs_RQ2.py",
    "gs_RQ3.py",
    "gs_RQ4.py",
    "gs_appendix_defense_tradeoff.py",
]

EXPECTED_OUTPUTS = [
    "rq1_final.png",
    "rq1_final.pdf",
    "rq2_six_panels.png",
    "rq2_six_panels.pdf",
    "rq3_camera_ready_adaptive_robustness.png",
    "rq3_camera_ready_adaptive_robustness.pdf",
    "rq4_camera_ready_tradeoff.png",
    "rq4_camera_ready_tradeoff.pdf",
    "defense_tradeoff_neg.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate camera-ready GrayShield RQ figures from release_results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "release_results",
        help="Directory containing rq1/, rq2/, rq3/, and rq4/ release artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "release_results" / "camera_ready_figures",
        help="Directory where generated figures will be copied.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the temporary working directory for debugging.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")


def prepare_workdir(workdir: Path, input_dir: Path) -> None:
    datasets = workdir / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    (workdir / "figures").mkdir(parents=True, exist_ok=True)
    (workdir / "images").mkdir(parents=True, exist_ok=True)

    inputs = {
        "rq1.jsonl": input_dir / "rq1" / "rq1.jsonl",
        "rq2.jsonl": input_dir / "rq2" / "rq2.jsonl",
        "rq3.jsonl": input_dir / "rq3" / "rq3.jsonl",
    }
    for name, source in inputs.items():
        require_file(source)
        shutil.copy2(source, datasets / name)


def run_script(script: str, workdir: Path) -> None:
    script_path = FIGURE_SCRIPTS / script
    require_file(script_path)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    print(f"[figures] running {script}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=workdir,
        env=env,
        check=True,
    )


def collect_outputs(workdir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for rel in EXPECTED_OUTPUTS:
        source = workdir / "figures" / rel
        if not source.exists():
            source = workdir / "images" / rel
        require_file(source)
        dest = output_dir / rel
        shutil.copy2(source, dest)
        copied.append(dest)
    return copied


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="grayshield-figures-"))
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="grayshield-figures-")
        workdir = Path(tmp.name)
        cleanup = True

    try:
        prepare_workdir(workdir, input_dir)
        for script in SCRIPTS:
            run_script(script, workdir)
        copied = collect_outputs(workdir, output_dir)
        print(f"[figures] wrote {len(copied)} files to {output_dir}")
        for path in copied:
            print(f"[figures]   {path.name}")
        if args.keep_workdir:
            print(f"[figures] kept workdir: {workdir}")
    finally:
        if cleanup:
            tmp.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
