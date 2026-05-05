#!/usr/bin/env python3
"""
Download malware samples from the GrayShield review dataset mirror.

For double-blind review, configure the dataset identifier through the
`GRAYSHIELD_HF_DATASET_ID` environment variable or edit the placeholder below.

Usage:
    python data/download_from_hf.py [--output-dir MALWARE_DIR] [--force]

Environment variables:
    HF_TOKEN: Optional Hugging Face access token
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HuggingFace dataset configuration
HF_DATASET_ID = os.getenv("GRAYSHIELD_HF_DATASET_ID", "<anonymous-dataset-id>")
HF_DATASET_SUBSET = "malware"  # Adjust if needed

# Get project root (parent of data directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "malware"


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        logger.error("Missing required package: huggingface_hub")
        logger.error("Install with: pip install huggingface-hub")
        return False


def download_from_huggingface(
    output_dir: Path,
    dataset_id: str = HF_DATASET_ID,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Download malware samples from HuggingFace dataset.

    Args:
        output_dir: Directory to save downloaded samples
        dataset_id: HuggingFace dataset ID
        force: Force re-download even if files exist

    Returns:
        Dictionary with download statistics
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from: {dataset_id}")
    logger.info(f"Output directory: {output_dir}")

    # Check for HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment. Public datasets only.")

    try:
        # List all files in the repository
        repo_files = list_repo_files(
            repo_id=dataset_id,
            repo_type="dataset",
            token=hf_token
        )

        # Filter for malware samples (adjust pattern as needed)
        malware_files = [
            f for f in repo_files
            if f.endswith('.malware') or 'malware/' in f
        ]

        if not malware_files:
            logger.warning(f"No malware files found in {dataset_id}")
            logger.info(f"Available files: {repo_files[:10]}")
            return {"error": "No malware files found", "total": 0}

        downloaded = []
        skipped = []
        failed = []

        for file_path in malware_files:
            filename = Path(file_path).name
            local_path = output_dir / filename

            # Skip if exists and not forcing
            if local_path.exists() and not force:
                logger.info(f"Skipping (exists): {filename}")
                skipped.append(filename)
                continue

            try:
                logger.info(f"Downloading: {filename}")
                downloaded_path = hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    filename=file_path,
                    local_dir=output_dir.parent,
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                downloaded.append(filename)
                logger.info(f"✓ Downloaded: {filename}")
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                failed.append({"file": filename, "error": str(e)})

        # Create manifest
        manifest = {
            "dataset_id": dataset_id,
            "downloaded": len(downloaded),
            "skipped": len(skipped),
            "failed": len(failed),
            "files": downloaded,
            "errors": failed
        }

        manifest_path = output_dir / "hf_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary:")
        logger.info(f"  Downloaded: {len(downloaded)} files")
        logger.info(f"  Skipped: {len(skipped)} files")
        logger.info(f"  Failed: {len(failed)} files")
        logger.info(f"  Manifest: {manifest_path}")
        logger.info(f"{'='*60}\n")

        return manifest

    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        return {"error": str(e), "total": 0}


def main():
    parser = argparse.ArgumentParser(
        description="Download malware samples from HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default directory (data/malware)
  python data/download_from_hf.py

  # Download to custom directory
  python data/download_from_hf.py --output-dir /path/to/malware

  # Force re-download all files
  python data/download_from_hf.py --force

Environment:
  HF_TOKEN    Optional HuggingFace token
        """
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default=HF_DATASET_ID,
        help=f"HuggingFace dataset ID (default: {HF_DATASET_ID})"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Security warning
    print(f"\n{'='*60}")
    print("⚠️  MALWARE DOWNLOAD WARNING ⚠️")
    print("Files are REAL MALWARE. Handle with EXTREME CAUTION.")
    print("- Use ONLY in isolated environments (VM, container)")
    print("- NEVER execute downloaded files")
    print("- For RESEARCH purposes only")
    print(f"{'='*60}\n")

    response = input("Continue with download? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        sys.exit(0)

    # Download
    result = download_from_huggingface(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        force=args.force
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
