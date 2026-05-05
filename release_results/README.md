# Anonymous Review Results Bundle

This directory is the anonymous review artifact bundle prepared for the
GrayShield paper.

## Provenance

- `rq1/` comes from `results/2026-03-28_1424_complete`.
- `rq2/`, `rq3/`, and `rq4/` come from `results/2026-03-21_0200_complete`.
- `rq234_experiment.log` preserves the shared source log for the merged
  `RQ2`-`RQ4` artifact set.
- `artifact_sources.json` records the exact source path for every copied
  artifact in this bundle.

## Why this merge exists

The review submission uses the refreshed RQ1 rerun together with the latest
completed RQ2--RQ4 results.  Keeping them under `release_results/` with one
subdirectory per research question lets reviewers inspect the exact artifacts
referenced in the manuscript without downloading the full scratch
`results/` workspace.

## Included artifact types

- `rq1/`, `rq2/`, `rq3/`, `rq4/` subdirectories with per-RQ artifacts
- shared provenance files at the top level
- paper tables in `rq3/`
- publication plots used in the manuscript and appendix
- source experiment logs

## Regeneration

Per-RQ plots and summary tables can be regenerated in place with:

```bash
PYTHONPATH=. python grayshield/visualization/rq1.py --input_dir release_results/rq1 --output_dir release_results/rq1
PYTHONPATH=. python grayshield/visualization/rq2.py --input_dir release_results/rq2 --output_dir release_results/rq2
PYTHONPATH=. python grayshield/visualization/rq3.py --input_dir release_results/rq3 --output_dir release_results/rq3
python scripts/generate_tables.py --output_dir release_results/rq3
```
