# GrayShield: Gray-Code-Guided Bit-Level Sanitization for Transformer Model Supply-Chain Security

This is the anonymized replication package for the NeurIPS 2026 submission
**"GrayShield: Gray-Code-Guided Bit-Level Sanitization for Transformer Model
Supply-Chain Security"**.

Author names, affiliations, personal repository links, and institutional
contact information are intentionally withheld for double-blind review.

## Anonymous Review Status

This repository is prepared as a double-blind review artifact.

- Git commit authors are anonymized as `Anonymous Authors <anonymous@example.com>`.
- Project metadata uses `Anonymous Authors` / `Anonymous Maintainers`.
- The artifact does not include author names, affiliations, personal emails,
  personal GitHub accounts, or institutional paths.
- The public repository does not ship real malware binaries.
- The only tracked payload files are benign byte payloads used for smoke tests.
- Exact paper payloads can be rehydrated by SHA256 from MalwareBazaar only by
  reviewers who have an approved isolated malware-analysis environment.

This layout follows the NeurIPS code/data review expectation that review
artifacts be anonymized. For the submission ZIP, include this repository
snapshot and do not include local files under `data/malware/` other than
`data/malware/.keep`.

## Anonymous Project Page

The anonymized project page for reviewers is:

https://grayshieldanonymous.netlify.app/

## Paper Summary

GrayShield studies an AI model supply-chain threat: malicious payloads hidden
in the least significant bits (LSBs) of FP32 transformer weights. The proposed
defense is a post-training, zero-data sanitization method that overwrites the
target mantissa LSBs with a secret-keyed Gray-code sequence.

The paper reports:

- 4 transformer presets: `bert_sst2`, `roberta_sentiment`, `vit_cifar10`,
  `swin_cifar10`.
- 2 main real-world MalwareBazaar payloads:
  - low-entropy JavaScript payload,
  - high-entropy executable payload.
- 7 defenses: `grayshield`, `pattern`, `random`, `gaussian`, `finetune`,
  `ptq`, `swp`.
- 5 attacker variants: `naive`, `interleave`, `repeat3`, `repeat5`, `rs`.
- Main operating depth: `x=19`.
- Main RQ3 result: GrayShield achieves `49.96% +/- 0.66` Recovery Reduction
  with sub-1% accuracy impact across completed attacker variants.

The formal claim is that, under a secret per-tensor seed and idealized keyed
PRF model, GrayShield reduces the effective covert-channel capacity to zero.

## Repository Contents

```text
grayshield/
  cli.py                         # CLI for RQ1-RQ4
  defense/                       # GrayShield and six baselines
  experiments/                   # Experiment runner
  lsb/                           # Bit-level injection/extraction
  metrics/                       # Payload/model/Pareto metrics
  models/                        # Model presets and target selection
  payload/                       # Payload encoders, RS code, MalwareBazaar helper
  visualization/                 # RQ plotting scripts

scripts/
  smoke_test.sh                  # Offline reviewer smoke test
  exps.sh                        # Main experiment wrapper
  experiments.sh                 # Core experiment grid runner
  generate_tables.py             # Table regeneration from bundled results
  generate_camera_ready_figures.py # Camera-ready RQ figure regeneration
  figures/                       # Standalone RQ/appendix plotting scripts

data/
  benign_payload.bin             # Benign smoke-test payload
  test_payload.bin               # Benign smoke-test payload
  download_paper_payloads.py     # Optional MalwareBazaar rehydration by SHA256
  download_from_hf.py            # Optional anonymous dataset mirror helper
  malware/.keep                  # Empty placeholder; real payloads are not tracked

release_results/
  rq1/ rq2/ rq3/ rq4/             # Curated paper result artifacts
  artifact_sources.json          # Artifact provenance
```

## Safety Notice

This repository is for defensive security research only.

- Do not execute malware payloads.
- Treat downloaded payloads as opaque byte strings only.
- Use an isolated VM/container if you rehydrate real malware samples.
- Expect antivirus tools to warn on rehydrated `.malware` files.
- Do not commit or upload rehydrated payloads.

The tracked repository contains no `.malware`, `.exe`, `.elf`, `.apk`,
`.xlsx`, `.zip`, `.dll`, `.so`, `.scr`, `.ps1`, or `.js` payload files.

## Installation

```bash
git clone https://github.com/anonymousgrayshield/grayshield.git
cd grayshield

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda is also fine:

```bash
conda create -n grayshield python=3.10
conda activate grayshield
pip install -r requirements.txt
```

## Reviewer Quick Path

### 1. Offline smoke test, no malware, no model download

This is the fastest command for reviewers. It verifies imports, payload bit
encoding/decoding, LSB injection, GrayShield sanitization, table regeneration,
and the main experiment wrapper's dry-run path.

```bash
bash scripts/smoke_test.sh
```

Expected final line:

```text
[smoke] OK
```

### 2. Regenerate paper tables from bundled artifacts

```bash
tmpdir="$(mktemp -d)"
cp release_results/rq3/rq3.jsonl "$tmpdir/rq3.jsonl"
python scripts/generate_tables.py --output_dir "$tmpdir"
ls "$tmpdir"/table*_defense_*
```

This uses only files already included in `release_results/`.

### 3. Regenerate plots from bundled artifacts

```bash
PYTHONPATH=. python grayshield/visualization/rq1.py \
  --input_dir release_results/rq1 \
  --output_dir release_results/rq1

PYTHONPATH=. python grayshield/visualization/rq2.py \
  --input_dir release_results/rq2 \
  --output_dir release_results/rq2

PYTHONPATH=. python grayshield/visualization/rq3.py \
  --input_dir release_results/rq3 \
  --output_dir release_results/rq3

PYTHONPATH=. python grayshield/visualization/rq4.py \
  --input_dir release_results \
  --output_dir release_results/rq4
```

The commands above regenerate the repository's standard release-result plots.
To regenerate the compact camera-ready RQ figures and appendix trade-off plot
from the same bundled `release_results/` artifacts, run:

```bash
python scripts/generate_camera_ready_figures.py
ls release_results/camera_ready_figures
```

Expected outputs:

```text
rq1_final.png
rq1_final.pdf
rq2_six_panels.png
rq2_six_panels.pdf
rq3_camera_ready_adaptive_robustness.png
rq3_camera_ready_adaptive_robustness.pdf
rq4_camera_ready_tradeoff.png
rq4_camera_ready_tradeoff.pdf
defense_tradeoff_neg.png
```

The wrapper creates a temporary `datasets/` view of the bundled JSONL files,
runs the standalone scripts in `scripts/figures/`, and copies the generated
figures to `release_results/camera_ready_figures/`.  These are compact
camera-ready figure variants; the original release-result figures referenced
by the current manuscript remain under the per-RQ directories
(`release_results/rq1/`, `release_results/rq2/`, `release_results/rq3/`,
and `release_results/rq4/`).

### 4. Dry-run the main experiment wrapper

This prints the actual commands that would be executed, without downloading
models or running the full grid.

```bash
bash scripts/exps.sh \
  --phase main \
  --rq rq1 \
  --models bert_sst2 \
  --payloads data \
  --no-paper-payloads \
  --n-payloads 1 \
  --n-eval 8 \
  --batch-size 4 \
  --dry-run
```

### 5. Lightweight real run with benign payload

This runs an actual small RQ2 instance with a benign payload. It may download
the requested Hugging Face model and evaluation data if they are not already
cached.

```bash
python -m grayshield.cli rq2 \
  --model bert_sst2 \
  --task sst2 \
  --payload_path data/test_payload.bin \
  --x 4 \
  --mode encoder_only \
  --defense grayshield \
  --attacker_variant naive \
  --n_eval 8 \
  --batch_size 4 \
  --device cpu \
  --output_dir results/reviewer_rq2_smoke \
  --seed 42
```

## Paper Payload Rehydration

The completed paper experiments use two real MalwareBazaar payloads. They are
not redistributed in this public anonymous repository.

In an isolated malware-analysis environment, reviewers may rehydrate the exact
main-paper payloads by SHA256:

```bash
export MALWAREBAZAAR_API_KEY=<optional-api-key>
python data/download_paper_payloads.py
```

To include the four supplementary appendix payloads as well:

```bash
python data/download_paper_payloads.py --appendix
```

Dry-run mode prints the selected hashes without downloading:

```bash
python data/download_paper_payloads.py --dry-run
python data/download_paper_payloads.py --appendix --dry-run
```

Downloaded files are written locally under `data/malware/` and are ignored by
Git. Do not include them in the anonymous submission ZIP.

If an anonymous controlled dataset mirror is supplied with the submission
materials, reviewers may instead set:

```bash
export GRAYSHIELD_HF_DATASET_ID=<anonymous-dataset-id>
python data/download_from_hf.py
```

## Main Paper Rerun

After the two main paper payloads have been rehydrated into `data/malware/`,
run:

```bash
bash scripts/exps.sh \
  --phase main \
  --rq all \
  --output-dir results/$(date +%F_%H%M)_complete \
  --visualize
```

Notes:

- `--phase main` uses the four main paper model presets:
  `bert_sst2`, `roberta_sentiment`, `vit_cifar10`, `swin_cifar10`.
- The default paper depth for RQ2/RQ3/RQ4 is `x=19`.
- The default RQ3 attacker variants are
  `naive,repeat3,repeat5,interleave,rs`.
- `scripts/exps.sh` enables GrayShield V3 by default.
- Full reruns may take substantial GPU time and require model/dataset
  downloads.

## Single-Command Examples

These commands are intentionally small. They use `data/test_payload.bin`, not
the real paper malware payloads.

### RQ1: injection feasibility

```bash
python -m grayshield.cli rq1 \
  --model bert_sst2 \
  --task sst2 \
  --payload_path data/test_payload.bin \
  --x 4 \
  --mode encoder_only \
  --n_eval 8 \
  --batch_size 4 \
  --device cpu \
  --output_dir results/reviewer_rq1_smoke \
  --seed 42
```

### RQ2: defense effectiveness

```bash
python -m grayshield.cli rq2 \
  --model bert_sst2 \
  --task sst2 \
  --payload_path data/test_payload.bin \
  --x 4 \
  --mode encoder_only \
  --defense grayshield \
  --attacker_variant naive \
  --n_eval 8 \
  --batch_size 4 \
  --device cpu \
  --output_dir results/reviewer_rq2_smoke \
  --seed 42
```

### RQ3: adaptive attackers

This checks all five implemented attacker variants for one model and a compact
three-defense subset. On CPU it can take several minutes; use the offline
smoke test above for the fastest reviewer check.

```bash
python -m grayshield.cli rq3 \
  --model bert_sst2 \
  --task sst2 \
  --payload_path data/test_payload.bin \
  --x 4 \
  --mode encoder_only \
  --attacker_variants naive,repeat3,repeat5,interleave,rs \
  --defenses grayshield,pattern,ptq \
  --n_eval 8 \
  --batch_size 4 \
  --device cpu \
  --output_dir results/reviewer_rq3_smoke \
  --seed 42
```

### RQ4: aggregate existing RQ2/RQ3 results

```bash
python -m grayshield.cli rq4 \
  --model bert_sst2 \
  --task sst2 \
  --payload_path data/test_payload.bin \
  --x 4 \
  --results_dir release_results \
  --output_dir results/reviewer_rq4_from_release \
  --device cpu
```

## Experiment Interface

`scripts/exps.sh` wraps `scripts/experiments.sh`, exports `PYTHONPATH`,
generates `.grayshield_key` when absent, enables `GRAYSHIELD_V3=1`, and
optionally launches visualization scripts.

Common options:

| Option | Description | Default |
|--------|-------------|---------|
| `--rq` | `rq1`, `rq2`, `rq3`, `rq4`, `all` | `all` |
| `--phase` | `main` or `appendix` | `main` |
| `--models` | Comma-separated preset list | phase-dependent |
| `--payloads` | Payload directory | `data` |
| `--x-bits` | RQ2/RQ3/RQ4 LSB depth | `19` |
| `--attacker-variant` | RQ2 attacker | `naive` |
| `--attacker-variants` | RQ3 attackers | `naive,repeat3,repeat5,interleave,rs` |
| `--n-eval` | Evaluation samples | `2048` |
| `--batch-size` | Evaluation batch size | `16` |
| `--no-paper-payloads` | Use benign/local payloads instead of SHA256 paper set | off |
| `--visualize` | Run plotting after experiments | off |
| `--visualize-only` | Plot only | off |
| `--dry-run` | Print commands without execution | off |

## Implemented Defenses

| Defense | Type | Paper role |
|---------|------|------------|
| `grayshield` | Secret-keyed Gray-code LSB overwrite | Proposed method |
| `pattern` | Fixed-pattern LSB overwrite | Deterministic baseline |
| `random` | Bernoulli LSB flipping | Stochastic bit baseline |
| `gaussian` | Additive FP32 Gaussian noise | Continuous perturbation baseline |
| `finetune` | Lightweight clean-data fine-tuning | Data-dependent baseline |
| `ptq` | Symmetric INT8 quantize-dequantize projection | Quantization baseline |
| `swp` | Magnitude-ranked selective weight perturbation | Literature-inspired baseline |

## Implemented Attacker Variants

| Variant | Mechanism |
|---------|-----------|
| `naive` | Direct LSB substitution |
| `interleave` | Deterministic payload-bit permutation |
| `repeat3` | 3x repetition with majority decoding |
| `repeat5` | 5x repetition with majority decoding |
| `rs` | Chunked `RS(255,127)` byte-level coding |

## Bundled Results

The curated paper artifact bundle lives under `release_results/`:

- `release_results/rq1/`: refreshed RQ1 injection feasibility artifacts.
- `release_results/rq2/`: defense effectiveness artifacts.
- `release_results/rq3/`: adaptive-attacker robustness artifacts and tables.
- `release_results/rq4/`: Pareto/deployment trade-off artifacts.
- `release_results/artifact_sources.json`: provenance for copied artifacts.

These artifacts are included so reviewers can inspect and regenerate the
published plots/tables without running the full GPU experiment grid.

## Reproducibility Settings

The paper reports:

- run seed: `1`
- evaluation seed: `42`
- evaluation subset size: `512`
- software: Python 3.13, PyTorch 2.9.1, Transformers 5.0.0
- hardware: Ubuntu 22.04.5, 2 x Intel Xeon Gold 6430, 503 GiB RAM,
  2 x NVIDIA RTX PRO 6000 Blackwell GPUs with 96 GB VRAM each

The anonymous package supports Python 3.10+ for review-time reproduction.

## GrayShield Keying

Standalone CLI runs can set:

```bash
export GRAYSHIELD_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export GRAYSHIELD_V3=1
```

`scripts/exps.sh` automatically creates `.grayshield_key` if absent and
exports `GRAYSHIELD_V3=1`. The `.grayshield_key` file is ignored by Git and
must not be committed.

## Anonymous Citation

For double-blind review, cite the artifact as:

```bibtex
@misc{anonymous2026grayshield,
  title        = {GrayShield: Gray-Code-Guided Bit-Level Sanitization for Transformer Model Supply-Chain Security},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {Anonymized NeurIPS 2026 review artifact},
  howpublished = {\url{https://github.com/anonymousgrayshield/grayshield}}
}
```

## License

MIT License. See [LICENSE](LICENSE).
