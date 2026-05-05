---
license: mit
task_categories:
- other
language:
- en
tags:
- security
- malware
- ml-security
- supply-chain-security
- steganography
- research
pretty_name: GrayShield Malware Payloads
---

# GrayShield Malware Payloads

This card documents the optional controlled anonymous dataset mirror for the
GrayShield paper payloads. The public anonymous GitHub repository does not
ship these payload binaries; it only includes this documentation and downloader
scripts.

## Safety warning

The controlled dataset mirror, when provided, contains real malware samples.
Handle them only in isolated research environments.

- Do not execute any file in this dataset.
- Treat every sample as an opaque byte sequence.
- Keep the `.malware` suffix intact.
- Use the dataset only for defensive or academic security research.

## Paper split

The anonymous review artifact uses two representative payloads in the
completed main-paper experiments and keeps four additional samples for
appendix analyses.

### Main-paper payloads

| Paper role | Filename | Type | Size (KiB) | Entropy (bits/byte) |
|------------|----------|------|-----------:|--------------------:|
| Main, low entropy | `c37c0db9...1a88.js.malware` | `js` | 134.293 | 4.821544 |
| Main, high entropy | `5704fabd...96d0.exe.malware` | `exe` | 8971.383 | 7.946050 |

### Supplementary appendix payloads

| Filename | Type | Size (KiB) | Entropy (bits/byte) |
|----------|------|-----------:|--------------------:|
| `5b6787d5...b360.xlsx.malware` | `xlsx` | 1456.177 | 7.998446 |
| `1368f3a8...1964.elf.malware` | `elf` | 8506.312 | 6.112022 |
| `31f110e2...147f.zip.malware` | `zip` | 3645.034 | 7.974474 |
| `9d5b59d0...b46e.apk.malware` | `apk` | 7068.168 | 7.603352 |

If a controlled mirror is provided, its metadata file records the exact
filenames, hashes, sizes, entropies, and paper roles. The public GitHub
artifact instead rehydrates the exact payloads by SHA256 with
`data/download_paper_payloads.py`.

## Source and preprocessing

The payloads were sourced from MalwareBazaar for defensive research. In the
GrayShield artifact they are never executed; the codebase only:

1. reads raw bytes,
2. converts bytes to bits for injection experiments,
3. measures recovery after post-training defenses.

## How to reproduce the paper payload setup

The public anonymous GitHub repository does not ship the executable payload
binaries. If approved for your review environment and an anonymous dataset
identifier is provided with the review package, rehydrate them from the
anonymous dataset mirror with:

```bash
export GRAYSHIELD_HF_DATASET_ID=<anonymous-dataset-id>
python data/download_from_hf.py
```

The script downloads the dataset into `data/malware/` and preserves the
metadata manifest used by the paper.

Repository: `https://github.com/anonymousgrayshield/grayshield`
Paper artifact: `https://github.com/anonymousgrayshield/grayshield`

## Citation

If you use this dataset, please cite the GrayShield paper and reference this
dataset URL.
