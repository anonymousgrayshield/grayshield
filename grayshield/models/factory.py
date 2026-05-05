"""
Model Factory for GrayShield

Provides model presets for various transformer architectures:
- Text models: BERT, DistilBERT, RoBERTa
- Vision models: ViT, Swin Transformer

Each preset maps to a specific HuggingFace model with task-appropriate training.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Literal, Tuple
import os
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
)
from ..config import ModelPreset

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Auto-Selection
# =============================================================================

def auto_select_gpu(min_free_gb: float = 2.0) -> Tuple[str, Dict[str, float]]:
    """
    Automatically select the GPU with the most free memory.

    Args:
        min_free_gb: Minimum required free memory in GB

    Returns:
        Tuple of (device_string, gpu_info_dict)
        device_string: e.g., "cuda:0" or "cpu" if no suitable GPU
        gpu_info_dict: {"gpu_index": int, "free_gb": float, "total_gb": float}

    Raises:
        RuntimeError: If no GPU meets the minimum free memory requirement
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu", {"gpu_index": -1, "free_gb": 0.0, "total_gb": 0.0}

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No CUDA devices found, falling back to CPU")
        return "cpu", {"gpu_index": -1, "free_gb": 0.0, "total_gb": 0.0}

    best_gpu = -1
    best_free = -1.0
    gpu_stats = []

    logger.info(f"Scanning {num_gpus} GPU(s) for auto-selection...")

    for i in range(num_gpus):
        try:
            free, total = torch.cuda.mem_get_info(i)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            gpu_stats.append((i, free_gb, total_gb))
            logger.info(f"  GPU {i}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")

            if free > best_free:
                best_free = free
                best_gpu = i
        except Exception as e:
            logger.warning(f"  GPU {i}: Error querying memory: {e}")

    best_free_gb = best_free / (1024 ** 3) if best_free > 0 else 0.0

    if best_gpu < 0 or best_free_gb < min_free_gb:
        raise RuntimeError(
            f"No GPU meets min_free_gb={min_free_gb}. "
            f"Best available: GPU {best_gpu} with {best_free_gb:.2f} GB free. "
            f"Consider: (1) Closing other GPU processes, "
            f"(2) Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
            f"(3) Using a smaller model."
        )

    device_str = f"cuda:{best_gpu}"
    total_gb = gpu_stats[best_gpu][2] if best_gpu < len(gpu_stats) else 0.0

    logger.info(f"Auto-selected: {device_str} ({best_free_gb:.2f} GB free)")

    # Suggest memory optimization if fragmentation might be an issue
    if best_free_gb < 4.0:
        logger.info(
            "TIP: If OOM occurs, try setting environment variable: "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )

    return device_str, {
        "gpu_index": best_gpu,
        "free_gb": best_free_gb,
        "total_gb": total_gb,
    }


def resolve_device(device: str, min_free_gb: float = 2.0) -> str:
    """
    Resolve device string, handling 'auto' and 'cuda:auto' specially.

    Args:
        device: Device string ("cpu", "cuda", "cuda:0", "cuda:auto", "auto")
        min_free_gb: Minimum free memory for auto-selection

    Returns:
        Resolved device string (e.g., "cuda:0" or "cpu")
    """
    if device in ("auto", "cuda:auto"):
        resolved, _ = auto_select_gpu(min_free_gb)
        return resolved
    elif device == "cuda":
        # "cuda" without index - auto-select best GPU
        resolved, _ = auto_select_gpu(min_free_gb)
        return resolved
    else:
        return device

# =============================================================================
# Model Presets
# =============================================================================

# Text Models - Sentiment/Classification tasks
TEXT_MODELS: Dict[str, ModelPreset] = {
    # BERT family - IMDB sentiment
    "bert_imdb": ModelPreset(
        name="bert_imdb",
        task_type="text",
        hf_model_id="textattack/bert-base-uncased-imdb",
    ),
    # BERT family - SST-2 sentiment
    "bert_sst2": ModelPreset(
        name="bert_sst2",
        task_type="text",
        hf_model_id="textattack/bert-base-uncased-SST-2",
    ),
    # DistilBERT - SST-2 (lightweight)
    "distilbert_sst2": ModelPreset(
        name="distilbert_sst2",
        task_type="text",
        hf_model_id="distilbert-base-uncased-finetuned-sst-2-english",
    ),
    # RoBERTa - SST-2 sentiment (2-class: negative/positive)
    # Note: Previous model "cardiffnlp/twitter-roberta-base-sentiment-latest" was 3-class
    # which caused ~48.7% accuracy on SST-2 (label mismatch)
    "roberta_sentiment": ModelPreset(
        name="roberta_sentiment",
        task_type="text",
        hf_model_id="textattack/roberta-base-SST-2",
    ),
}

# Vision Models - Image classification tasks
VISION_MODELS: Dict[str, ModelPreset] = {
    # Vision Transformer - CIFAR-10
    "vit_cifar10": ModelPreset(
        name="vit_cifar10",
        task_type="vision",
        hf_model_id="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    ),
    # Swin Transformer - CIFAR-10
    "swin_cifar10": ModelPreset(
        name="swin_cifar10",
        task_type="vision",
        hf_model_id="Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10",
    ),
    # Original ImageNet presets (kept for backward compatibility)
    "vit_imagenet": ModelPreset(
        name="vit_imagenet",
        task_type="vision",
        hf_model_id="google/vit-base-patch16-224",
    ),
    "swin_imagenet": ModelPreset(
        name="swin_imagenet",
        task_type="vision",
        hf_model_id="microsoft/swin-base-patch4-window7-224",
    ),
}

# Combined presets dictionary
PRESETS: Dict[str, ModelPreset] = {**TEXT_MODELS, **VISION_MODELS}

# Dataset compatibility mapping
DATASET_COMPATIBILITY: Dict[str, str] = {
    # Text models
    "bert_imdb": "imdb",
    "bert_sst2": "sst2",
    "distilbert_sst2": "sst2",
    "roberta_sentiment": "sst2",  # Use SST-2 for evaluation
    # Vision models
    "vit_cifar10": "cifar10",
    "swin_cifar10": "cifar10",
    "vit_imagenet": "cifar10",  # Use CIFAR-10 for evaluation
    "swin_imagenet": "cifar10",
}


def get_compatible_task(preset_name: str) -> str:
    """Get the compatible task/dataset for a given preset."""
    return DATASET_COMPATIBILITY.get(preset_name, "sst2")


def list_presets(task_type: Optional[Literal["text", "vision", "all"]] = None) -> List[str]:
    """
    List available model presets.

    Args:
        task_type: Filter by task type ("text", "vision", or "all")

    Returns:
        List of preset names
    """
    if task_type == "text":
        return list(TEXT_MODELS.keys())
    elif task_type == "vision":
        return list(VISION_MODELS.keys())
    else:
        return list(PRESETS.keys())


def get_preset_info(preset_name: str) -> Dict:
    """Get detailed information about a preset."""
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    preset = PRESETS[preset_name]
    return {
        "name": preset.name,
        "task_type": preset.task_type,
        "hf_model_id": preset.hf_model_id,
        "compatible_dataset": DATASET_COMPATIBILITY.get(preset_name, "unknown"),
    }


def load_preset(
    preset_name: str,
    device: str = "cuda",
    min_free_gb: float = 2.0,
):
    """
    Load a model preset from HuggingFace.

    Args:
        preset_name: Name of the preset (e.g., "bert_sst2", "vit_cifar10")
        device: Device to load model on. Options:
            - "cuda" or "cuda:auto": Auto-select GPU with most free memory
            - "cuda:0", "cuda:1", etc.: Use specific GPU
            - "cpu": Use CPU
            - "auto": Auto-select best available device
        min_free_gb: Minimum free GPU memory required for auto-selection

    Returns:
        Tuple of (preset, model, processor)

    Raises:
        KeyError: If preset_name is not found
        RuntimeError: If no suitable GPU is available (when using auto-selection)
    """
    if preset_name not in PRESETS:
        available = list(PRESETS.keys())
        raise KeyError(
            f"Unknown preset: {preset_name}.\n"
            f"Available text models: {list(TEXT_MODELS.keys())}\n"
            f"Available vision models: {list(VISION_MODELS.keys())}"
        )

    preset = PRESETS[preset_name]
    model_id = preset.hf_model_id
    proc_id = preset.hf_processor_id or model_id

    if preset.task_type == "text":
        processor = AutoTokenizer.from_pretrained(proc_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        processor = AutoImageProcessor.from_pretrained(proc_id, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_id)

    model.eval()

    # Resolve device with auto-selection support
    target_device = resolve_device(device, min_free_gb)
    model.to(target_device)

    logger.info(f"Model {preset_name} loaded on {target_device}")

    return preset, model, processor


def load_all_text_models(device: str = "cuda") -> Dict[str, tuple]:
    """Load all text models."""
    return {name: load_preset(name, device) for name in TEXT_MODELS}


def load_all_vision_models(device: str = "cuda") -> Dict[str, tuple]:
    """Load all vision models."""
    return {name: load_preset(name, device) for name in VISION_MODELS}
