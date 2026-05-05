# Payload Files

## Safety Notice

This directory contains opaque byte payloads used to evaluate LSB-based
payload injection and post-training sanitization in GrayShield.

- The repository never executes or interprets these files.
- The public anonymous GitHub artifact does not include executable malware
  binaries.  It includes benign payloads for smoke testing plus metadata for
  the paper payload set.
- Use them only in isolated research environments.

The complete paper payload bundle uses real MalwareBazaar samples. Rehydrate
those files only by SHA256 from MalwareBazaar or through the controlled
anonymous dataset mirror described below, and only inside an isolated research
environment.

## What the paper uses

The completed main-paper experiments use two representative MalwareBazaar
payloads chosen to span the low- and high-entropy extremes of the current
threat model:

| Paper role | Filename | Type | Size (KiB) | Entropy (bits/byte) |
|------------|----------|------|-----------:|--------------------:|
| Main, low entropy | `c37c0db9...1a88.js.malware` | `js` | 134.293 | 4.821544 |
| Main, high entropy | `5704fabd...96d0.exe.malware` | `exe` | 8971.383 | 7.946050 |

The remaining four metadata entries are supplementary payloads used for
appendix analyses and future extensions:

| Paper role | Filename | Type | Size (KiB) | Entropy (bits/byte) |
|------------|----------|------|-----------:|--------------------:|
| Appendix | `5b6787d5...b360.xlsx.malware` | `xlsx` | 1456.177 | 7.998446 |
| Appendix | `1368f3a8...1964.elf.malware` | `elf` | 8506.312 | 6.112022 |
| Appendix | `31f110e2...147f.zip.malware` | `zip` | 3645.034 | 7.974474 |
| Appendix | `9d5b59d0...b46e.apk.malware` | `apk` | 7068.168 | 7.603352 |

Two additional local files are only for smoke tests and quick validation:

- `benign_payload.bin`
- `test_payload.bin`

## Directory contents

```text
data/
|-- README.md
|-- HF_DATASET_CARD.md
|-- benign_payload.bin
|-- test_payload.bin
|-- download_paper_payloads.py
|-- download_from_hf.py
`-- malware/
    |-- .keep
    |-- metadata.json
    `-- manifest.json
```

- `malware/metadata.json` is the paper-facing metadata file used for the
  anonymous dataset card and reproducibility notes.
- `malware/manifest.json` contains MalwareBazaar download metadata for the
  subset fetched directly from the source API.

## Preferred: rehydrate from MalwareBazaar by SHA256

The preferred reviewer path avoids redistributing malware while preserving
exact reproducibility. In an isolated VM/container, run:

```bash
export MALWAREBAZAAR_API_KEY=<optional-api-key>
python data/download_paper_payloads.py
```

This downloads and verifies the two main-paper payloads by SHA256. To also
download the supplementary appendix payloads:

```bash
python data/download_paper_payloads.py --appendix
```

The script writes `.malware` files into `data/malware/` and records a local
`paper_payload_manifest.json`. The experiment scripts then discover the
main-paper payloads automatically through their SHA256 hashes.

## Optional controlled dataset mirror

The curated payload set can be mirrored at an anonymous dataset URL supplied
with the review materials:

`<anonymous-dataset-url>`

If approved for your review environment, rehydrate the payload bundle with:

```bash
python data/download_from_hf.py
```

To place the files elsewhere:

```bash
python data/download_from_hf.py --output-dir /path/to/malware
```

`HF_TOKEN` is optional for public mirrors.  For double-blind review, set
`GRAYSHIELD_HF_DATASET_ID` to the anonymous dataset identifier before running
the download script.

## Handling rules

- Never execute the payload files.
- Treat all samples as raw bytes only.
- Keep the `.malware` suffix intact.
- Expect anti-virus software to quarantine or warn on these files.
- Use them strictly for defensive security research.

## Citation

If you use the public payload set, please cite the paper and reference the
dataset URL above.
