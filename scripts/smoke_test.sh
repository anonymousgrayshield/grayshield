#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MALWAREBAZAAR_API_KEY="${MALWAREBAZAAR_API_KEY:-reviewer-smoke-placeholder}"

echo "[smoke] CLI import"
python -m grayshield.cli --help >/dev/null

echo "[smoke] Bit-level injection and GrayShield defense"
python - <<'PY'
import copy
import torch

from grayshield.defense.gray_code import GrayShieldDefense
from grayshield.lsb.stego import inject_bits, extract_bits
from grayshield.metrics.payload import bit_accuracy, exact_recovery
from grayshield.payload.encoding import encode_payload, decode_payload, AttackerVariant
from grayshield.payload.loader import load_payload_bits


torch.manual_seed(42)

model = torch.nn.Sequential(
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
).float()
target_names = ["0.weight", "2.weight"]

payload = load_payload_bits("data/test_payload.bin", max_bits=512)
encoded, enc_report = encode_payload(payload.bits, AttackerVariant.NAIVE)

poisoned = copy.deepcopy(model)
report = inject_bits(poisoned, target_names, encoded, x=4)
assert report.written_bits_total == len(encoded), report

recovered = extract_bits(poisoned, target_names, x=4, n_bits=len(encoded))
decoded, dec_report = decode_payload(recovered, AttackerVariant.NAIVE, original_length=len(payload.bits))
assert exact_recovery(payload.bits, decoded), "payload should recover exactly before defense"

defended = copy.deepcopy(poisoned)
GrayShieldDefense().apply(
    defended,
    target_names,
    x=4,
    seed=123,
    use_v2=True,
    use_v3=True,
    secret_key=b"anonymous-review-smoke-key",
)
sanitized = extract_bits(defended, target_names, x=4, n_bits=len(encoded))
decoded_sanitized, _ = decode_payload(
    sanitized,
    AttackerVariant.NAIVE,
    original_length=len(payload.bits),
)
post_acc = bit_accuracy(payload.bits, decoded_sanitized)
assert post_acc < 0.95, f"sanitization did not perturb payload enough: {post_acc:.3f}"

print(f"pre_defense_bitacc=1.000 post_defense_bitacc={post_acc:.3f}")
print(f"encoding={enc_report.variant} decoded_bits={dec_report.decoded_bits}")
PY

echo "[smoke] Table regeneration from bundled result artifact"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cp release_results/rq3/rq3.jsonl "$tmpdir/rq3.jsonl"
python scripts/generate_tables.py --output_dir "$tmpdir" >/dev/null
test -s "$tmpdir/table1_defense_comparison.md"
test -s "$tmpdir/table2_attacker_robustness.md"

echo "[smoke] Dry-run reviewer command path"
bash scripts/exps.sh \
  --phase main \
  --rq rq1 \
  --models bert_sst2 \
  --payloads data \
  --no-paper-payloads \
  --n-payloads 1 \
  --n-eval 8 \
  --batch-size 4 \
  --dry-run >/dev/null

echo "[smoke] OK"
